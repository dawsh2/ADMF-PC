"""In-memory event storage."""

from typing import Dict, Any, Optional, List, Set
from collections import deque, defaultdict
from datetime import datetime
import json

from ..protocols import EventStorageProtocol
from ..types import Event

class MemoryEventStorage(EventStorageProtocol):
    """
    In-memory event storage with configurable retention.
    
    Used by EventTracer for temporary storage during execution.
    Supports multiple indices for efficient querying.
    """
    
    def __init__(self, max_size: Optional[int] = None, enable_indices: bool = True):
        """
        Initialize memory storage.
        
        Args:
            max_size: Maximum events to store (None for unlimited)
            enable_indices: Whether to maintain indices for fast queries
        """
        self.max_size = max_size
        self.enable_indices = enable_indices
        
        # Primary storage
        self.events: deque = deque(maxlen=max_size) if max_size else deque()
        
        # Indices for fast lookup
        if enable_indices:
            self._event_id_index: Dict[str, Event] = {}
            self._correlation_index: Dict[str, List[Event]] = defaultdict(list)
            self._type_index: Dict[str, List[Event]] = defaultdict(list)
        
        self._total_stored = 0
    
    def store(self, event: Event) -> None:
        """Store an event."""
        # Check if we're at capacity with no maxlen
        if self.max_size is None and len(self.events) >= 1000000:  # 1M safety limit
            # Remove oldest
            oldest = self.events.popleft()
            self._remove_from_indices(oldest)
        
        # Store event
        self.events.append(event)
        self._total_stored += 1
        
        # Update indices
        if self.enable_indices:
            self._add_to_indices(event)
    
    def retrieve(self, event_id: str) -> Optional[Event]:
        """Retrieve event by ID."""
        if self.enable_indices:
            return self._event_id_index.get(event_id)
        
        # Linear search if no indices
        for event in self.events:
            if event.metadata.get('event_id') == event_id:
                return event
        return None
    
    def query(self, criteria: Dict[str, Any]) -> List[Event]:
        """Query events by criteria."""
        # Fast path for indexed queries
        if self.enable_indices:
            if 'correlation_id' in criteria:
                return self._correlation_index.get(criteria['correlation_id'], []).copy()
            
            if 'event_type' in criteria:
                return self._type_index.get(criteria['event_type'], []).copy()
        
        # General query
        results = []
        for event in self.events:
            if self._matches_criteria(event, criteria):
                results.append(event)
        
        return results
    
    def prune(self, criteria: Dict[str, Any]) -> int:
        """Prune events matching criteria."""
        # Special handling for correlation-based pruning
        if 'correlation_id' in criteria and self.enable_indices:
            correlation_id = criteria['correlation_id']
            events_to_prune = self._correlation_index.get(correlation_id, []).copy()
            
            # Check for exclusions
            if 'exclude_event_id' in criteria:
                exclude_id = criteria['exclude_event_id']
                events_to_prune = [e for e in events_to_prune 
                                 if e.metadata.get('event_id') != exclude_id]
            
            # Remove from storage
            pruned = 0
            for event in events_to_prune:
                try:
                    self.events.remove(event)
                    self._remove_from_indices(event)
                    pruned += 1
                except ValueError:
                    pass  # Already removed
            
            return pruned
        
        # General pruning
        to_remove = []
        for event in self.events:
            if self._matches_criteria(event, criteria):
                to_remove.append(event)
        
        for event in to_remove:
            self.events.remove(event)
            self._remove_from_indices(event)
        
        return len(to_remove)
    
    def count(self) -> int:
        """Get total event count."""
        return len(self.events)
    
    def export_to_file(self, filepath: str) -> None:
        """Export all events to file."""
        with open(filepath, 'w') as f:
            for event in self.events:
                f.write(json.dumps(event.to_dict(), default=str) + '\n')
    
    def clear(self) -> None:
        """Clear all events."""
        self.events.clear()
        if self.enable_indices:
            self._event_id_index.clear()
            self._correlation_index.clear()
            self._type_index.clear()
    
    def prune_oldest(self, count: int) -> int:
        """Prune oldest events."""
        pruned = 0
        for _ in range(min(count, len(self.events))):
            event = self.events.popleft()
            self._remove_from_indices(event)
            pruned += 1
        return pruned
    
    # Private methods
    
    def _add_to_indices(self, event: Event) -> None:
        """Add event to indices."""
        # Event ID index
        event_id = event.metadata.get('event_id')
        if event_id:
            self._event_id_index[event_id] = event
        
        # Correlation index
        if event.correlation_id:
            self._correlation_index[event.correlation_id].append(event)
        
        # Type index
        event_type = event.event_type
        self._type_index[event_type].append(event)
    
    def _remove_from_indices(self, event: Event) -> None:
        """Remove event from indices."""
        if not self.enable_indices:
            return
        
        # Event ID index
        event_id = event.metadata.get('event_id')
        if event_id and event_id in self._event_id_index:
            del self._event_id_index[event_id]
        
        # Correlation index
        if event.correlation_id and event.correlation_id in self._correlation_index:
            try:
                self._correlation_index[event.correlation_id].remove(event)
                if not self._correlation_index[event.correlation_id]:
                    del self._correlation_index[event.correlation_id]
            except ValueError:
                pass
        
        # Type index
        if event.event_type in self._type_index:
            try:
                self._type_index[event.event_type].remove(event)
                if not self._type_index[event.event_type]:
                    del self._type_index[event.event_type]
            except ValueError:
                pass
    
    def _matches_criteria(self, event: Event, criteria: Dict[str, Any]) -> bool:
        """Check if event matches criteria."""
        for key, value in criteria.items():
            if key.startswith('exclude_'):
                continue
                
            if hasattr(event, key):
                if getattr(event, key) != value:
                    return False
            elif key in event.metadata:
                if event.metadata[key] != value:
                    return False
            elif key in event.payload:
                if event.payload[key] != value:
                    return False
            else:
                return False
        
        return True