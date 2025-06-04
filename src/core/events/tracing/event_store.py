"""
Event store for persistent storage of traced events.

This module provides persistent storage capabilities for event tracing,
enabling long-term analysis and pattern discovery through the data mining
architecture.
"""

import logging
from typing import List, Dict, Any, Optional, Protocol, Iterator
from datetime import datetime, timedelta
from pathlib import Path
import json
from abc import abstractmethod
from dataclasses import asdict
import gzip

from .traced_event import TracedEvent
from ..semantic import SemanticEvent

logger = logging.getLogger(__name__)


class StorageBackend(Protocol):
    """Protocol for event storage backends."""
    
    @abstractmethod
    def write_events(self, events: List[TracedEvent]) -> None:
        """Write batch of events to storage."""
        ...
    
    @abstractmethod
    def read_events(self, 
                   correlation_id: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   event_types: Optional[List[str]] = None) -> Iterator[TracedEvent]:
        """Read events from storage with filtering."""
        ...
    
    @abstractmethod
    def get_correlation_ids(self, 
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> List[str]:
        """Get all correlation IDs in time range."""
        ...
    
    @abstractmethod
    def close(self) -> None:
        """Close storage connection."""
        ...


class EventStore:
    """
    Persistent storage for traced events with intelligent batching.
    
    This class manages:
    - Event batching for efficient writes
    - Compression for long-term storage
    - Multiple storage backend support
    - Correlation-based indexing
    """
    
    def __init__(self, 
                 backend: StorageBackend,
                 batch_size: int = 1000,
                 flush_interval: float = 5.0):
        """
        Initialize event store.
        
        Args:
            backend: Storage backend implementation
            batch_size: Number of events to batch before writing
            flush_interval: Max seconds between flushes
        """
        self.backend = backend
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Batching
        self.event_batch: List[TracedEvent] = []
        self.last_flush = datetime.now()
        
        # Statistics
        self.events_written = 0
        self.batches_written = 0
        
        logger.info(f"EventStore initialized with {type(backend).__name__} backend")
    
    def store_event(self, event: TracedEvent) -> None:
        """Store single event (batched for efficiency)."""
        self.event_batch.append(event)
        
        # Check if we should flush
        if (len(self.event_batch) >= self.batch_size or 
            (datetime.now() - self.last_flush).total_seconds() > self.flush_interval):
            self.flush()
    
    def store_events(self, events: List[TracedEvent]) -> None:
        """Store multiple events."""
        for event in events:
            self.store_event(event)
    
    def flush(self) -> None:
        """Force flush of pending events."""
        if not self.event_batch:
            return
        
        try:
            self.backend.write_events(self.event_batch)
            self.events_written += len(self.event_batch)
            self.batches_written += 1
            
            logger.debug(f"Flushed {len(self.event_batch)} events to storage")
            
            self.event_batch.clear()
            self.last_flush = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to flush events: {e}")
            raise
    
    def load_events_for_correlation(self, correlation_id: str) -> List[TracedEvent]:
        """Load all events for a specific correlation ID."""
        events = list(self.backend.read_events(correlation_id=correlation_id))
        logger.info(f"Loaded {len(events)} events for correlation {correlation_id}")
        return events
    
    def load_events_in_range(self, 
                            start_time: datetime,
                            end_time: datetime,
                            event_types: Optional[List[str]] = None) -> Iterator[TracedEvent]:
        """Load events in time range with optional type filtering."""
        return self.backend.read_events(
            start_time=start_time,
            end_time=end_time,
            event_types=event_types
        )
    
    def find_optimization_runs(self,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> List[str]:
        """Find all optimization run correlation IDs."""
        return self.backend.get_correlation_ids(start_time, end_time)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            'events_written': self.events_written,
            'batches_written': self.batches_written,
            'pending_events': len(self.event_batch),
            'backend': type(self.backend).__name__
        }
    
    def close(self) -> None:
        """Close storage, flushing any pending events."""
        if self.event_batch:
            self.flush()
        self.backend.close()
        logger.info(f"EventStore closed. Total events written: {self.events_written}")
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure proper cleanup."""
        self.close()


class InMemoryBackend(StorageBackend):
    """
    In-memory storage backend for testing and short-term analysis.
    
    WARNING: This backend loses all data on process exit!
    Use only for testing or when persistence is not required.
    """
    
    def __init__(self):
        """Initialize in-memory storage."""
        self.events: List[TracedEvent] = []
        self.correlation_index: Dict[str, List[int]] = {}
        logger.warning("InMemoryBackend initialized - data will not persist!")
    
    def write_events(self, events: List[TracedEvent]) -> None:
        """Store events in memory."""
        start_idx = len(self.events)
        
        for i, event in enumerate(events):
            idx = start_idx + i
            self.events.append(event)
            
            # Update correlation index
            if event.correlation_id not in self.correlation_index:
                self.correlation_index[event.correlation_id] = []
            self.correlation_index[event.correlation_id].append(idx)
    
    def read_events(self,
                   correlation_id: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   event_types: Optional[List[str]] = None) -> Iterator[TracedEvent]:
        """Read events with filtering."""
        # Get candidate indices
        if correlation_id:
            indices = self.correlation_index.get(correlation_id, [])
            candidates = [self.events[i] for i in indices]
        else:
            candidates = self.events
        
        # Apply filters
        for event in candidates:
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if event_types and event.event_type not in event_types:
                continue
            yield event
    
    def get_correlation_ids(self,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> List[str]:
        """Get unique correlation IDs."""
        if not start_time and not end_time:
            return list(self.correlation_index.keys())
        
        # Filter by time
        correlation_ids = set()
        for event in self.events:
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            correlation_ids.add(event.correlation_id)
        
        return list(correlation_ids)
    
    def close(self) -> None:
        """No-op for in-memory backend."""
        pass


def create_event_store(backend_type: str = "memory", **kwargs) -> EventStore:
    """
    Factory function to create event store with specified backend.
    
    Args:
        backend_type: Type of backend ("memory", "parquet", "timescale")
        **kwargs: Backend-specific configuration
        
    Returns:
        Configured EventStore instance
    """
    if backend_type == "memory":
        backend = InMemoryBackend()
    elif backend_type == "parquet":
        # Import here to avoid circular dependencies
        from .storage_backends import ParquetBackend
        backend = ParquetBackend(**kwargs)
    elif backend_type == "timescale":
        from .storage_backends import TimescaleBackend
        backend = TimescaleBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
    
    return EventStore(backend, **kwargs)