"""
Unified Event and Execution Tracer for ADMF-PC

This module consolidates ExecutionTracer and EventTracer into a single
comprehensive tracing system that provides:
- Event tracking with correlation IDs and causation chains
- Flow verification with trace points and canonical implementation checks
- Lightweight and comprehensive modes
- Performance metrics and pattern discovery
"""

import logging
import time
from typing import Dict, List, Optional, Deque, Union, Any
from collections import defaultdict, deque
import uuid
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import json
import os

from ...types.events import Event, EventType
from .traced_event import TracedEvent

logger = logging.getLogger(__name__)


class TracePoint(str, Enum):
    """Key trace points in the execution flow."""
    DATA_LOAD = "data.load"
    FEATURE_CALC = "feature.calc"
    SIGNAL_GEN = "signal.gen"
    ORDER_CREATE = "order.create"
    ORDER_ROUTE = "order.route"
    ORDER_EXEC = "order.exec"
    FILL_CREATE = "fill.create"
    FILL_ROUTE = "fill.route"
    PORTFOLIO_UPDATE = "portfolio.update"


@dataclass
class TraceEntry:
    """Lightweight trace entry for flow verification."""
    timestamp: float
    trace_point: TracePoint
    component: str  # Which file/class is executing
    details: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    event_id: Optional[str] = None  # Links to full event if available


class TracingMode(str, Enum):
    """Tracing modes for different use cases."""
    LIGHTWEIGHT = "lightweight"  # Just trace points (like ExecutionTracer)
    COMPREHENSIVE = "comprehensive"  # Full events (like EventTracer)
    HYBRID = "hybrid"  # Both trace points and full events


class UnifiedTracer:
    """
    Unified tracer combining ExecutionTracer and EventTracer functionality.
    
    This tracer provides:
    - Lightweight flow tracing with trace points
    - Comprehensive event tracing with full data
    - Canonical implementation verification
    - Performance metrics and analysis
    """
    
    def __init__(
        self,
        correlation_id: Optional[str] = None,
        max_events: int = 10000,
        mode: TracingMode = TracingMode.HYBRID,
        enabled: bool = True,
        trace_file_path: Optional[str] = None
    ):
        """
        Initialize the unified tracer.
        
        Args:
            correlation_id: Optional correlation ID for this trace session
            max_events: Maximum number of events to keep in memory
            mode: Tracing mode (lightweight, comprehensive, or hybrid)
            enabled: Whether tracing is enabled
            trace_file_path: Optional path to write trace data to file
        """
        self.correlation_id = correlation_id or self._generate_correlation_id()
        self.max_events = max_events
        self.mode = mode
        self.enabled = enabled
        self.trace_file_path = trace_file_path
        self._trace_file = None
        
        # Lightweight tracing (from ExecutionTracer)
        self.trace_entries: List[TraceEntry] = []
        self._trace_counter = 0
        
        # Comprehensive tracing (from EventTracer)
        self.traced_events: Deque[TracedEvent] = deque(maxlen=max_events)
        self.event_index: Dict[str, TracedEvent] = {}
        self.sequence_counter = 0
        
        # Statistics
        self.event_counts = defaultdict(int)
        self.container_counts = defaultdict(int)
        self.trace_point_counts = defaultdict(int)
        
        # Canonical component mapping for verification
        self.canonical_components = {
            TracePoint.DATA_LOAD: "csv_handler.py",
            TracePoint.FEATURE_CALC: "symbol_timeframe_container.py",
            TracePoint.SIGNAL_GEN: ["stateless_momentum.py", "momentum_strategy.py"],
            TracePoint.ORDER_CREATE: "portfolio_container.py",
            TracePoint.ORDER_EXEC: "execution_container.py",
            TracePoint.PORTFOLIO_UPDATE: "portfolio_container.py"
        }
        
        logger.info(f"UnifiedTracer initialized: mode={mode.value}, correlation_id={self.correlation_id}")
        
        # Open trace file if path provided
        if self.trace_file_path and self.enabled:
            self._open_trace_file()
    
    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID for this session."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        return f"workflow_{timestamp}_{unique_id}"
    
    def trace_point(
        self,
        trace_point: TracePoint,
        component: str,
        details: Optional[Dict[str, Any]] = None,
        event: Optional[Event] = None
    ) -> str:
        """
        Add a lightweight trace point (ExecutionTracer functionality).
        
        Args:
            trace_point: The trace point enum
            component: Component/file executing
            details: Optional details
            event: Optional event associated with this trace point
            
        Returns:
            Event ID for correlation
        """
        if not self.enabled or self.mode == TracingMode.COMPREHENSIVE:
            return ""
        
        # Generate event ID
        self._trace_counter += 1
        event_id = f"trace_{self._trace_counter:04d}"
        
        # Create trace entry
        entry = TraceEntry(
            timestamp=time.time(),
            trace_point=trace_point,
            component=component,
            details=details or {},
            correlation_id=self.correlation_id,
            event_id=event_id if event else None
        )
        
        self.trace_entries.append(entry)
        self.trace_point_counts[trace_point] += 1
        
        # Write to file if enabled
        self._write_to_file('trace_point', {
            'trace_point': trace_point.value,
            'component': component,
            'details': details,
            'correlation_id': self.correlation_id,
            'event_id': event_id
        })
        
        # Log the trace
        logger.debug(f"TRACE[{self.correlation_id}] {trace_point.value} in {component}: {details}")
        
        # If we have an associated event and we're in hybrid mode, trace it too
        if event and self.mode == TracingMode.HYBRID:
            self.trace_event(event, component)
        
        return event_id
    
    def trace_event(self, event: Event, source_container: str) -> TracedEvent:
        """
        Trace a full event (EventTracer functionality).
        
        Args:
            event: The event to trace
            source_container: Container that emitted the event
            
        Returns:
            TracedEvent with full metadata
        """
        if not self.enabled or self.mode == TracingMode.LIGHTWEIGHT:
            return None
        
        self.sequence_counter += 1
        
        # Generate event ID if not present
        event_id = event.metadata.get('event_id')
        if not event_id:
            event_id = f"{event.event_type.value}_{uuid.uuid4().hex[:8]}"
        
        # Extract causation ID
        causation_id = event.metadata.get('causation_id', '')
        
        # Create traced event
        traced = TracedEvent(
            event_id=event_id,
            event_type=event.event_type.value,
            timestamp=event.timestamp,
            correlation_id=self.correlation_id,
            causation_id=causation_id,
            source_container=source_container,
            created_at=datetime.now(),
            data=event.payload,
            metadata=event.metadata,
            sequence_num=self.sequence_counter
        )
        
        # Store in memory
        self.traced_events.append(traced)
        self.event_index[event_id] = traced
        
        # Update statistics
        self.event_counts[event.event_type] += 1
        self.container_counts[source_container] += 1
        
        # Write to file if enabled
        self._write_to_file('event', {
            'event_id': traced.event_id,
            'event_type': traced.event_type,
            'source_container': source_container,
            'correlation_id': traced.correlation_id,
            'causation_id': traced.causation_id,
            'sequence_num': traced.sequence_num,
            'data': traced.data
        })
        
        return traced
    
    def get_flow_for_correlation(self, correlation_id: Optional[str] = None) -> List[TraceEntry]:
        """Get trace flow for a correlation ID (ExecutionTracer functionality)."""
        target_id = correlation_id or self.correlation_id
        return [t for t in self.trace_entries if t.correlation_id == target_id]
    
    def get_events_by_type(self, event_type: Union[EventType, str]) -> List[TracedEvent]:
        """Get all events of a specific type (EventTracer functionality)."""
        type_str = event_type.value if hasattr(event_type, 'value') else str(event_type)
        return [e for e in self.traced_events if e.event_type == type_str]
    
    def verify_canonical_flow(self) -> Dict[str, Any]:
        """
        Verify that execution followed canonical implementations.
        
        Returns:
            Dict with violations and statistics
        """
        violations = []
        
        for trace in self.trace_entries:
            expected = self.canonical_components.get(trace.trace_point)
            if expected:
                # Handle both string and list of acceptable components
                acceptable = expected if isinstance(expected, list) else [expected]
                if not any(comp in trace.component for comp in acceptable):
                    violations.append({
                        'trace_point': trace.trace_point.value,
                        'expected': expected,
                        'actual': trace.component,
                        'correlation_id': trace.correlation_id,
                        'timestamp': trace.timestamp
                    })
        
        return {
            'violations': violations,
            'total_traces': len(self.trace_entries),
            'canonical_compliance': 1.0 - (len(violations) / len(self.trace_entries)) if self.trace_entries else 1.0
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of tracing data.
        
        Returns:
            Dict with statistics and analysis
        """
        summary = {
            'correlation_id': self.correlation_id,
            'mode': self.mode.value,
            'trace_points': {
                'total': len(self.trace_entries),
                'by_type': dict(self.trace_point_counts)
            },
            'events': {
                'total': len(self.traced_events),
                'by_type': dict(self.event_counts),
                'by_container': dict(self.container_counts)
            }
        }
        
        # Add canonical flow verification
        if self.trace_entries:
            summary['canonical_verification'] = self.verify_canonical_flow()
        
        # Add performance metrics if we have events
        if self.traced_events:
            first_event = self.traced_events[0]
            last_event = self.traced_events[-1]
            duration = (last_event.created_at - first_event.created_at).total_seconds()
            
            summary['performance'] = {
                'duration_seconds': duration,
                'events_per_second': len(self.traced_events) / duration if duration > 0 else 0,
                'average_sequence_gap': duration / len(self.traced_events) if self.traced_events else 0
            }
        
        return summary
    
    def get_event_chain(self, event_id: str) -> List[TracedEvent]:
        """
        Get the causation chain for an event.
        
        Args:
            event_id: Starting event ID
            
        Returns:
            List of events in the causation chain
        """
        chain = []
        current_event = self.event_index.get(event_id)
        
        # Walk backwards through causation
        while current_event:
            chain.append(current_event)
            if current_event.causation_id:
                current_event = self.event_index.get(current_event.causation_id)
            else:
                break
        
        return list(reversed(chain))
    
    def clear(self):
        """Clear all tracing data."""
        self.trace_entries.clear()
        self.traced_events.clear()
        self.event_index.clear()
        self.event_counts.clear()
        self.container_counts.clear()
        self.trace_point_counts.clear()
        self._trace_counter = 0
        self.sequence_counter = 0
        
        logger.info(f"UnifiedTracer cleared for correlation_id: {self.correlation_id}")
    
    def _open_trace_file(self):
        """Open trace file for writing."""
        try:
            # Create directory if needed
            if self.trace_file_path:
                os.makedirs(os.path.dirname(self.trace_file_path), exist_ok=True)
                self._trace_file = open(self.trace_file_path, 'w')
                # Write header
                header = {
                    'correlation_id': self.correlation_id,
                    'mode': self.mode.value,
                    'start_time': datetime.now().isoformat(),
                    'max_events': self.max_events
                }
                self._trace_file.write(json.dumps({'header': header}) + '\n')
                logger.info(f"Opened trace file: {self.trace_file_path}")
        except Exception as e:
            logger.error(f"Failed to open trace file {self.trace_file_path}: {e}")
            self._trace_file = None
    
    def _write_to_file(self, entry_type: str, data: Dict[str, Any]):
        """Write entry to trace file if open."""
        if self._trace_file and not self._trace_file.closed:
            try:
                entry = {
                    'type': entry_type,
                    'timestamp': time.time(),
                    'data': data
                }
                self._trace_file.write(json.dumps(entry) + '\n')
                self._trace_file.flush()  # Ensure data is written
            except Exception as e:
                logger.error(f"Failed to write to trace file: {e}")
    
    def close(self):
        """Close trace file and clean up."""
        if self._trace_file and not self._trace_file.closed:
            try:
                # Write summary before closing
                summary = self.get_summary()
                self._write_to_file('summary', summary)
                self._trace_file.close()
                logger.info(f"Closed trace file: {self.trace_file_path}")
            except Exception as e:
                logger.error(f"Error closing trace file: {e}")
        self._trace_file = None


# Convenience factory functions

def create_lightweight_tracer(correlation_id: Optional[str] = None) -> UnifiedTracer:
    """Create a tracer optimized for lightweight flow tracking."""
    return UnifiedTracer(
        correlation_id=correlation_id,
        mode=TracingMode.LIGHTWEIGHT,
        max_events=1000  # Smaller buffer for lightweight mode
    )


def create_comprehensive_tracer(correlation_id: Optional[str] = None) -> UnifiedTracer:
    """Create a tracer optimized for full event tracking."""
    return UnifiedTracer(
        correlation_id=correlation_id,
        mode=TracingMode.COMPREHENSIVE,
        max_events=100000  # Larger buffer for comprehensive mode
    )


def create_hybrid_tracer(correlation_id: Optional[str] = None) -> UnifiedTracer:
    """Create a tracer that does both flow tracking and event tracing."""
    return UnifiedTracer(
        correlation_id=correlation_id,
        mode=TracingMode.HYBRID,
        max_events=50000  # Medium buffer for hybrid mode
    )


def create_portfolio_tracer(portfolio_id: str, trace_dir: str = "./traces") -> UnifiedTracer:
    """Create a tracer for a specific portfolio with file output."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    trace_file_path = os.path.join(trace_dir, f"{portfolio_id}_{timestamp}.jsonl")
    
    return UnifiedTracer(
        correlation_id=f"portfolio_{portfolio_id}_{timestamp}",
        mode=TracingMode.HYBRID,
        max_events=10000,  # Smaller buffer since writing to file
        trace_file_path=trace_file_path
    )