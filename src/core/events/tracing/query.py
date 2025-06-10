"""Query interface for event analysis and data mining."""

from typing import List, Dict, Any, Optional, Iterator
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from ..protocols import EventStorageProtocol
from ..types import Event

class EventQueryInterface:
    """
    Query interface for analyzing traced events.
    
    Enables data mining and pattern detection on event streams.
    """
    
    def __init__(self, storage_backend: EventStorageProtocol):
        self.storage = storage_backend
        
    def query_by_time_range(self, start: datetime, end: datetime, 
                           event_types: Optional[List[str]] = None) -> pd.DataFrame:
        """Query events within time range, return as DataFrame."""
        events = self.storage.query({
            'start_time': start,
            'end_time': end,
            'event_types': event_types
        })
        return self._events_to_dataframe(events)
        
    def find_patterns(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find event patterns for strategy development."""
        # Example: Find all cases where signal → fill took > 100ms
        results = []
        correlation_groups = self._group_by_correlation()
        
        for correlation_id, events in correlation_groups.items():
            if self._matches_pattern(events, pattern):
                results.append({
                    'correlation_id': correlation_id,
                    'events': events,
                    'metrics': self._calculate_pattern_metrics(events, pattern)
                })
        return results
        
    def analyze_performance(self, container_id: str) -> Dict[str, Any]:
        """Analyze performance metrics from event traces."""
        events = self.storage.query({'container_id': container_id})
        
        return {
            'event_latencies': self._calculate_latencies(events),
            'throughput': self._calculate_throughput(events),
            'bottlenecks': self._identify_bottlenecks(events)
        }
        
    def export_to_parquet(self, output_path: Path, 
                         filters: Optional[Dict[str, Any]] = None) -> None:
        """Export filtered events to Parquet for data science workflows."""
        events = self.storage.query(filters or {})
        df = self._events_to_dataframe(events)
        df.to_parquet(output_path, compression='snappy')
    
    def _events_to_dataframe(self, events: List[Event]) -> pd.DataFrame:
        """Convert events to pandas DataFrame."""
        records = []
        for event in events:
            record = {
                'timestamp': event.timestamp,
                'event_type': event.event_type,
                'container_id': event.container_id,
                'source_id': event.source_id,
                'correlation_id': event.correlation_id,
                'causation_id': event.causation_id,
            }
            # Flatten payload
            for key, value in event.payload.items():
                record[f'payload_{key}'] = value
            # Flatten metadata
            for key, value in event.metadata.items():
                record[f'metadata_{key}'] = value
            records.append(record)
            
        return pd.DataFrame(records)
    
    def _group_by_correlation(self) -> Dict[str, List[Event]]:
        """Group all events by correlation ID."""
        # This is a simplified version - real implementation would be more efficient
        all_events = self.storage.query({})
        groups = defaultdict(list)
        
        for event in all_events:
            if event.correlation_id:
                groups[event.correlation_id].append(event)
                
        return dict(groups)
    
    def _matches_pattern(self, events: List[Event], pattern: Dict[str, Any]) -> bool:
        """Check if event sequence matches pattern."""
        # Simplified pattern matching
        if 'sequence' in pattern:
            event_types = [e.event_type for e in events]
            return all(event_type in event_types for event_type in pattern['sequence'])
        return True
    
    def _calculate_pattern_metrics(self, events: List[Event], pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for a pattern match."""
        if not events:
            return {}
            
        return {
            'event_count': len(events),
            'duration': (events[-1].timestamp - events[0].timestamp).total_seconds(),
            'event_types': list(set(e.event_type for e in events))
        }
    
    def _calculate_latencies(self, events: List[Event]) -> Dict[str, float]:
        """Calculate event processing latencies."""
        latencies = defaultdict(list)
        
        for event in events:
            if 'delivered_at' in event.metadata and event.timestamp:
                delivered = datetime.fromisoformat(event.metadata['delivered_at'])
                latency = (delivered - event.timestamp).total_seconds() * 1000  # ms
                latencies[event.event_type].append(latency)
                
        # Calculate averages
        return {
            event_type: sum(times) / len(times) if times else 0
            for event_type, times in latencies.items()
        }
    
    def _calculate_throughput(self, events: List[Event]) -> Dict[str, float]:
        """Calculate event throughput per second."""
        if not events:
            return {}
            
        # Sort by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        duration = (sorted_events[-1].timestamp - sorted_events[0].timestamp).total_seconds()
        
        if duration == 0:
            return {}
            
        # Count by type
        type_counts = defaultdict(int)
        for event in events:
            type_counts[event.event_type] += 1
            
        return {
            event_type: count / duration
            for event_type, count in type_counts.items()
        }
    
    def _identify_bottlenecks(self, events: List[Event]) -> List[Dict[str, Any]]:
        """Identify processing bottlenecks from event patterns."""
        bottlenecks = []
        
        # Simple example: Find events with high latency
        for event in events:
            if 'delivered_at' in event.metadata and event.timestamp:
                delivered = datetime.fromisoformat(event.metadata['delivered_at'])
                latency = (delivered - event.timestamp).total_seconds() * 1000
                
                if latency > 100:  # More than 100ms
                    bottlenecks.append({
                        'event_id': event.metadata.get('event_id'),
                        'event_type': event.event_type,
                        'latency_ms': latency,
                        'timestamp': event.timestamp
                    })
                    
        return bottlenecks