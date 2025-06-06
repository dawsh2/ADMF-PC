"""
Query Interface for Event Mining

This module provides tools for querying and analyzing traced events,
supporting the data mining architecture described in docs/architecture/data-mining-*.
"""

import json
import os
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from collections import defaultdict
import pandas as pd

from .unified_tracer import UnifiedTracer, TraceEntry, TracedEvent
from ...types.events import EventType


class TraceQuery:
    """
    Query interface for mining event traces.
    
    Supports:
    - Loading traces from files or memory
    - Filtering by event type, container, time range
    - Pattern detection and anomaly identification
    - Performance analysis and bottleneck detection
    """
    
    def __init__(self, source: Union[str, UnifiedTracer, List[str]]):
        """
        Initialize query interface.
        
        Args:
            source: Can be:
                - Path to trace file
                - UnifiedTracer instance (for in-memory traces)
                - List of trace file paths (for multi-file analysis)
        """
        self.traces: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.trace_points: List[Dict[str, Any]] = []
        
        if isinstance(source, str):
            # Load from single file
            self._load_from_file(source)
        elif isinstance(source, list):
            # Load from multiple files
            for file_path in source:
                self._load_from_file(file_path)
        elif isinstance(source, UnifiedTracer):
            # Load from tracer instance
            self._load_from_tracer(source)
        else:
            raise ValueError(f"Invalid source type: {type(source)}")
    
    def _load_from_file(self, file_path: str):
        """Load traces from a JSONL file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Trace file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entry_type = entry.get('type')
                    
                    if entry_type == 'event':
                        self.events.append(entry['data'])
                    elif entry_type == 'trace_point':
                        self.trace_points.append(entry['data'])
                    elif entry_type == 'header':
                        # Store header info if needed
                        pass
                    elif entry_type == 'summary':
                        # Store summary if needed
                        pass
                except json.JSONDecodeError:
                    continue
    
    def _load_from_tracer(self, tracer: UnifiedTracer):
        """Load traces from a UnifiedTracer instance."""
        # Load events
        for event in tracer.traced_events:
            self.events.append({
                'event_id': event.event_id,
                'event_type': event.event_type,
                'source_container': event.source_container,
                'correlation_id': event.correlation_id,
                'causation_id': event.causation_id,
                'sequence_num': event.sequence_num,
                'timestamp': event.timestamp,
                'data': event.data
            })
        
        # Load trace points
        for trace in tracer.trace_entries:
            self.trace_points.append({
                'trace_point': trace.trace_point.value,
                'component': trace.component,
                'details': trace.details,
                'correlation_id': trace.correlation_id,
                'timestamp': trace.timestamp
            })
    
    # Query Methods
    
    def filter_events(
        self,
        event_type: Optional[Union[EventType, str]] = None,
        source_container: Optional[str] = None,
        correlation_id: Optional[str] = None,
        time_range: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """Filter events by various criteria."""
        results = self.events
        
        if event_type:
            type_str = event_type.value if hasattr(event_type, 'value') else str(event_type)
            results = [e for e in results if e.get('event_type') == type_str]
        
        if source_container:
            results = [e for e in results if e.get('source_container') == source_container]
        
        if correlation_id:
            results = [e for e in results if e.get('correlation_id') == correlation_id]
        
        if time_range:
            start, end = time_range
            results = [e for e in results if start <= e.get('timestamp', 0) <= end]
        
        return results
    
    def get_event_chain(self, event_id: str) -> List[Dict[str, Any]]:
        """Get the causation chain for an event."""
        chain = []
        event_index = {e['event_id']: e for e in self.events}
        
        current = event_index.get(event_id)
        
        # Walk backwards through causation
        while current:
            chain.append(current)
            if current.get('causation_id'):
                current = event_index.get(current['causation_id'])
            else:
                break
        
        return list(reversed(chain))
    
    def get_container_flow(self, container_id: str) -> List[Dict[str, Any]]:
        """Get all events from a specific container in order."""
        events = [e for e in self.events if e.get('source_container') == container_id]
        return sorted(events, key=lambda e: e.get('sequence_num', 0))
    
    def get_signal_to_fill_paths(self) -> List[List[Dict[str, Any]]]:
        """Trace paths from signals to fills."""
        paths = []
        
        # Find all SIGNAL events
        signals = self.filter_events(event_type=EventType.SIGNAL)
        
        for signal in signals:
            path = [signal]
            
            # Find ORDER events caused by this signal
            orders = [e for e in self.events 
                     if e.get('event_type') == 'ORDER' 
                     and e.get('causation_id') == signal['event_id']]
            
            for order in orders:
                order_path = path + [order]
                
                # Find FILL events caused by this order
                fills = [e for e in self.events 
                        if e.get('event_type') == 'FILL' 
                        and e.get('causation_id') == order['event_id']]
                
                for fill in fills:
                    paths.append(order_path + [fill])
        
        return paths
    
    def analyze_latencies(self) -> Dict[str, Dict[str, float]]:
        """Analyze latencies between event types."""
        latencies = defaultdict(list)
        
        # Group events by correlation ID
        by_correlation = defaultdict(list)
        for event in self.events:
            by_correlation[event.get('correlation_id', '')].append(event)
        
        # Calculate latencies
        for correlation_id, events in by_correlation.items():
            events = sorted(events, key=lambda e: e.get('timestamp', 0))
            
            for i in range(len(events) - 1):
                current = events[i]
                next_event = events[i + 1]
                
                if current.get('timestamp') and next_event.get('timestamp'):
                    latency = next_event['timestamp'] - current['timestamp']
                    key = f"{current['event_type']} -> {next_event['event_type']}"
                    latencies[key].append(latency)
        
        # Calculate statistics
        stats = {}
        for key, values in latencies.items():
            if values:
                stats[key] = {
                    'count': len(values),
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
        
        return stats
    
    def detect_anomalies(self, threshold_stddev: float = 3.0) -> List[Dict[str, Any]]:
        """Detect anomalous events based on latency patterns."""
        anomalies = []
        
        # Get latency statistics
        latency_stats = self.analyze_latencies()
        
        # Check each event transition
        for i in range(len(self.events) - 1):
            current = self.events[i]
            next_event = self.events[i + 1]
            
            if current.get('correlation_id') == next_event.get('correlation_id'):
                key = f"{current['event_type']} -> {next_event['event_type']}"
                
                if key in latency_stats:
                    stats = latency_stats[key]
                    if stats['count'] > 10:  # Need enough samples
                        latency = next_event['timestamp'] - current['timestamp']
                        mean = stats['mean']
                        
                        # Simple anomaly detection
                        if latency > mean * threshold_stddev:
                            anomalies.append({
                                'type': 'high_latency',
                                'from_event': current,
                                'to_event': next_event,
                                'latency': latency,
                                'expected': mean,
                                'factor': latency / mean
                            })
        
        return anomalies
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert events to pandas DataFrame for analysis."""
        if not self.events:
            return pd.DataFrame()
        
        # Flatten event data
        rows = []
        for event in self.events:
            row = {
                'event_id': event.get('event_id'),
                'event_type': event.get('event_type'),
                'source_container': event.get('source_container'),
                'correlation_id': event.get('correlation_id'),
                'causation_id': event.get('causation_id'),
                'sequence_num': event.get('sequence_num'),
                'timestamp': event.get('timestamp')
            }
            
            # Add selected data fields
            data = event.get('data', {})
            if isinstance(data, dict):
                row['symbol'] = data.get('symbol')
                row['direction'] = data.get('direction')
                row['quantity'] = data.get('quantity')
                row['price'] = data.get('price')
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the trace data."""
        df = self.to_dataframe()
        
        summary = {
            'total_events': len(self.events),
            'total_trace_points': len(self.trace_points),
            'event_types': df['event_type'].value_counts().to_dict() if not df.empty else {},
            'containers': df['source_container'].value_counts().to_dict() if not df.empty else {},
            'latency_stats': self.analyze_latencies()
        }
        
        # Add signal to fill conversion rate
        signal_paths = self.get_signal_to_fill_paths()
        if signal_paths:
            summary['signal_to_fill_rate'] = len(signal_paths) / len(self.filter_events(event_type=EventType.SIGNAL))
        
        return summary


# Convenience functions

def query_trace_file(file_path: str) -> TraceQuery:
    """Create a query interface for a single trace file."""
    return TraceQuery(file_path)


def query_trace_directory(directory: str, pattern: str = "*.jsonl") -> TraceQuery:
    """Create a query interface for all trace files in a directory."""
    import glob
    files = glob.glob(os.path.join(directory, pattern))
    return TraceQuery(files)


def query_portfolio_traces(portfolio_id: str, trace_dir: str = "./traces") -> TraceQuery:
    """Query all traces for a specific portfolio."""
    import glob
    pattern = f"{portfolio_id}_*.jsonl"
    files = glob.glob(os.path.join(trace_dir, pattern))
    return TraceQuery(files)