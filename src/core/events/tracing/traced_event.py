"""
Traced Event Structure for Event Lineage and Performance Tracking

This module provides the TracedEvent dataclass which adds comprehensive
tracking metadata to events, enabling:
- Complete event lineage (causation chains)
- Performance profiling (latency tracking)
- Pattern discovery and analysis
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
import uuid


@dataclass
class TracedEvent:
    """
    Event with full lineage and performance tracking.
    
    This enhanced event structure enables:
    1. Event lineage tracking (correlation and causation)
    2. Performance monitoring (latency at each stage)
    3. Pattern discovery (via SQL analytics)
    4. Debugging complex event flows
    
    Attributes:
        event_id: Unique identifier for this event
        event_type: Type of event (SIGNAL, ORDER, FILL, etc.)
        timestamp: When the event logically occurred
        correlation_id: Groups all events in a backtest/session
        causation_id: ID of the event that caused this one
        source_container: Name of container that emitted event
        created_at: When event object was created
        emitted_at: When event was published to bus
        received_at: When event was received by handler
        processed_at: When event processing completed
        data: Event payload
        version: Event schema version
        sequence_number: Order within correlation group
        partition_key: For future sharding/partitioning
    """
    
    # Identity
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}")
    event_type: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Lineage
    correlation_id: str = ""  # Groups all events in a backtest
    causation_id: str = ""    # ID of event that caused this one
    source_container: str = "" # Container that emitted event
    
    # Performance tracking
    created_at: datetime = field(default_factory=datetime.now)
    emitted_at: Optional[datetime] = None
    received_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    
    # Payload
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    version: str = "1.0"
    sequence_number: int = 0
    partition_key: str = ""
    
    @property
    def latency_ms(self) -> float:
        """Total processing latency in milliseconds"""
        if self.processed_at and self.created_at:
            return (self.processed_at - self.created_at).total_seconds() * 1000
        return 0.0
        
    @property
    def queue_time_ms(self) -> float:
        """Time spent waiting in event bus queue"""
        if self.received_at and self.emitted_at:
            return (self.received_at - self.emitted_at).total_seconds() * 1000
        return 0.0
        
    @property
    def processing_time_ms(self) -> float:
        """Time spent in actual event processing"""
        if self.processed_at and self.received_at:
            return (self.processed_at - self.received_at).total_seconds() * 1000
        return 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'correlation_id': self.correlation_id,
            'causation_id': self.causation_id,
            'source_container': self.source_container,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'emitted_at': self.emitted_at.isoformat() if self.emitted_at else None,
            'received_at': self.received_at.isoformat() if self.received_at else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'data': self.data,
            'version': self.version,
            'sequence_number': self.sequence_number,
            'partition_key': self.partition_key,
            'latency_ms': self.latency_ms,
            'queue_time_ms': self.queue_time_ms,
            'processing_time_ms': self.processing_time_ms
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TracedEvent':
        """Create from dictionary (for loading from storage)"""
        # Parse datetime strings
        for field in ['timestamp', 'created_at', 'emitted_at', 'received_at', 'processed_at']:
            if field in data and data[field] and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
                
        # Remove calculated fields
        data.pop('latency_ms', None)
        data.pop('queue_time_ms', None)
        data.pop('processing_time_ms', None)
        
        return cls(**data)