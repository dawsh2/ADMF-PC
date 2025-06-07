"""
Event Tracer for tracking event flow through the system

This module provides the EventTracer class which:
- Generates correlation IDs for tracking related events
- Maintains in-memory index for fast event lookup
- Tracks event sequences and causation chains
- Enables pattern discovery and performance analysis
"""

import logging
from typing import Dict, List, Optional, Deque, Any
from collections import defaultdict, deque
import uuid
from datetime import datetime
import json
import os

from ...types.events import Event

logger = logging.getLogger(__name__)


class EventTracer:
    """
    Traces event flow through the system.
    
    This class is responsible for:
    1. Converting regular events to traced events with full metadata
    2. Maintaining correlation IDs for related event groups
    3. Tracking event sequences and causation chains
    4. Providing fast in-memory lookup of recent events
    
    The tracer keeps a limited history in memory for performance,
    with options to persist to storage for deeper analysis.
    """
    
    def __init__(self, correlation_id: Optional[str] = None, max_events: int = 10000):
        """
        Initialize the event tracer.
        
        Args:
            correlation_id: Optional correlation ID for this trace session.
                           If not provided, generates a unique ID.
            max_events: Maximum number of events to keep in memory (default 10,000)
        """
        self.correlation_id = correlation_id or self._generate_correlation_id()
        self.max_events = max_events
        
        # Keep last N events in memory for fast access
        self.traced_events: Deque[Event] = deque(maxlen=max_events)
        
        # Index for O(1) lookup by event_id
        self.event_index: Dict[str, Event] = {}
        
        # Track sequences within this correlation
        self.sequence_counter = 0
        
        # Statistics tracking
        self.event_counts = defaultdict(int)
        self.container_counts = defaultdict(int)
        
        logger.info(f"EventTracer initialized with correlation_id: {self.correlation_id}")
        
    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID for this backtest/session run"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        return f"backtest_{timestamp}_{unique_id}"
        
    def trace_event(self, event: Event, source_container: str) -> Event:
        """
        Enhance event with tracking metadata and store for analysis.
        
        Args:
            event: The event to trace
            source_container: Name of the container that emitted the event
            
        Returns:
            The same event with enhanced metadata
        """
        self.sequence_counter += 1
        
        # Ensure event has an ID
        event_id = event.metadata.get('event_id')
        if not event_id:
            event_id = f"{event.event_type.value if hasattr(event.event_type, 'value') else event.event_type}_{uuid.uuid4().hex[:8]}"
            event.metadata['event_id'] = event_id
        
        # Set correlation_id if not already set
        if not event.correlation_id:
            event.correlation_id = self.correlation_id
        
        # Set sequence number
        if not event.sequence_number:
            event.sequence_number = self.sequence_counter
        
        # Enhance metadata for tracing
        event.metadata['trace_info'] = {
            'source_container': source_container,
            'traced_at': datetime.now().isoformat(),
            'sequence_in_correlation': self.sequence_counter
        }
        
        # Store in memory for quick access
        self._store_event(event)
        
        # Update statistics
        event_type = event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)
        self.event_counts[event_type] += 1
        self.container_counts[source_container] += 1
        
        return event
        
    def _store_event(self, event: Event):
        """Store event in memory with index management"""
        # Add to deque (automatically removes oldest if at capacity)
        if len(self.traced_events) == self.max_events:
            # Remove oldest event from index
            oldest = self.traced_events[0]
            self.event_index.pop(oldest.metadata.get('event_id', ''), None)
            
        self.traced_events.append(event)
        self.event_index[event.metadata['event_id']] = event
        
    def get_event(self, event_id: str) -> Optional[Event]:
        """Get traced event by ID with O(1) lookup"""
        return self.event_index.get(event_id)
        
    def find_events_by_type(self, event_type: str) -> List[Event]:
        """Find all events of a given type"""
        return [e for e in self.traced_events if 
                (e.event_type.value if hasattr(e.event_type, 'value') else str(e.event_type)) == event_type]
        
    def find_events_by_container(self, container: str) -> List[Event]:
        """Find all events from a specific container"""
        return [e for e in self.traced_events 
                if e.metadata.get('trace_info', {}).get('source_container') == container]
        
    def trace_causation_chain(self, event_id: str) -> List[Event]:
        """
        Trace the complete causation chain for an event.
        
        Returns both ancestors (what caused this) and descendants (what this caused).
        """
        event = self.get_event(event_id)
        if not event:
            return []
            
        # Find ancestors (events that led to this one)
        ancestors = self._find_ancestors(event)
        
        # Find descendants (events caused by this one)
        descendants = self._find_descendants(event)
        
        # Return complete chain in chronological order
        return ancestors + [event] + descendants
        
    def _find_ancestors(self, event: Event) -> List[Event]:
        """Find all events that led to this event"""
        ancestors = []
        current = event
        
        while current.causation_id:
            parent = self.get_event(current.causation_id)
            if not parent or parent in ancestors:
                break
            ancestors.insert(0, parent)  # Insert at beginning for chronological order
            current = parent
            
        return ancestors
        
    def _find_descendants(self, event: Event) -> List[Event]:
        """Find all events caused by this event"""
        descendants = []
        event_id = event.metadata.get('event_id', '')
        to_check = [event_id]
        checked = set()
        
        while to_check:
            parent_id = to_check.pop(0)
            if parent_id in checked:
                continue
            checked.add(parent_id)
            
            # Find all events caused by this parent
            for traced in self.traced_events:
                if traced.causation_id == parent_id:
                    descendants.append(traced)
                    to_check.append(traced.metadata.get('event_id', ''))
                    
        return sorted(descendants, key=lambda e: e.sequence_number or 0)
        
    def calculate_latency_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate latency statistics by event type"""
        stats = defaultdict(lambda: {'count': 0, 'total': 0.0, 'max': 0.0, 'min': float('inf')})
        
        for event in self.traced_events:
            # Calculate latency from trace timing if available
            trace_timing = event.metadata.get('trace_timing', {})
            if 'emitted_at' in trace_timing and 'processed_at' in trace_timing:
                try:
                    emitted = datetime.fromisoformat(trace_timing['emitted_at'])
                    processed = datetime.fromisoformat(trace_timing['processed_at'])
                    latency_ms = (processed - emitted).total_seconds() * 1000
                    
                    event_type = event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)
                    event_stats = stats[event_type]
                    event_stats['count'] += 1
                    event_stats['total'] += latency_ms
                    event_stats['max'] = max(event_stats['max'], latency_ms)
                    event_stats['min'] = min(event_stats['min'], latency_ms)
                except:
                    pass  # Skip if timing data is invalid
                
        # Calculate averages
        result = {}
        for event_type, event_stats in stats.items():
            if event_stats['count'] > 0:
                result[event_type] = {
                    'avg_ms': event_stats['total'] / event_stats['count'],
                    'max_ms': event_stats['max'],
                    'min_ms': event_stats['min'],
                    'count': event_stats['count']
                }
                
        return result
        
    def get_summary(self) -> Dict[str, any]:
        """Get summary statistics of traced events"""
        return {
            'correlation_id': self.correlation_id,
            'total_events': len(self.traced_events),
            'event_counts': dict(self.event_counts),
            'container_counts': dict(self.container_counts),
            'latency_stats': self.calculate_latency_stats(),
            'sequence_range': {
                'first': self.traced_events[0].sequence_number if self.traced_events and self.traced_events[0].sequence_number else 0,
                'last': self.sequence_counter
            }
        }
        
    def clear(self):
        """Clear all traced events and reset counters"""
        self.traced_events.clear()
        self.event_index.clear()
        self.event_counts.clear()
        self.container_counts.clear()
        self.sequence_counter = 0
    
    def save_to_file(self, output_path: str):
        """
        Save traced events to file in JSON Lines format.
        
        Args:
            output_path: Path where to save the trace file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save events in JSON Lines format (one event per line)
        with open(output_path, 'w') as f:
            # First line: metadata
            metadata = {
                'correlation_id': self.correlation_id,
                'total_events': len(self.traced_events),
                'event_counts': dict(self.event_counts),
                'container_counts': dict(self.container_counts),
                'saved_at': datetime.now().isoformat()
            }
            f.write(json.dumps(metadata) + '\n')
            
            # Subsequent lines: events
            for event in self.traced_events:
                # Convert event to dict for serialization
                event_dict = {
                    'event_type': event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type),
                    'data': event.data,
                    'timestamp': event.timestamp,
                    'correlation_id': event.correlation_id,
                    'causation_id': event.causation_id,
                    'sequence_number': event.sequence_number,
                    'metadata': event.metadata
                }
                f.write(json.dumps(event_dict, default=str) + '\n')
        
        logger.info(f"Saved {len(self.traced_events)} events to {output_path}")
        
    def __len__(self) -> int:
        """Return number of traced events"""
        return len(self.traced_events)
        
    def __repr__(self) -> str:
        return f"EventTracer(correlation_id={self.correlation_id}, events={len(self)})"