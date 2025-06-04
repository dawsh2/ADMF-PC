"""
Storage backends for event persistence.
Supports TimescaleDB, Parquet, and in-memory storage with container isolation.
"""
import json
import logging
from typing import List, Dict, Any, Optional, Iterator, Protocol
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import gzip
from collections import defaultdict
from contextlib import contextmanager

from src.core.events.tracing.traced_event import TracedEvent


class StorageBackend(Protocol):
    """Protocol for event storage backends."""
    
    def write_events(self, events: List[TracedEvent]) -> None:
        """Write events to storage."""
        ...
        
    def read_events(
        self, 
        correlation_id: Optional[str] = None,
        source_container: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> Iterator[TracedEvent]:
        """Read events from storage with filtering."""
        ...
        
    def delete_events(self, correlation_id: str) -> int:
        """Delete events by correlation ID."""
        ...
        
    def get_correlation_ids(
        self,
        source_container: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[str]:
        """Get unique correlation IDs."""
        ...


class InMemoryBackend:
    """In-memory storage backend with container isolation support."""
    
    def __init__(self, max_events_per_container: int = 100000):
        self.max_events_per_container = max_events_per_container
        # Isolate events by source container
        self._events_by_container: Dict[str, List[TracedEvent]] = defaultdict(list)
        # Secondary index by correlation_id
        self._events_by_correlation: Dict[str, List[TracedEvent]] = defaultdict(list)
        self._lock = None  # TODO: Add threading.Lock if needed
        self.logger = logging.getLogger(f"{__name__}.InMemoryBackend")
        
    def write_events(self, events: List[TracedEvent]) -> None:
        """Write events maintaining container isolation."""
        for event in events:
            # Store by container (isolation)
            container_events = self._events_by_container[event.source_container]
            container_events.append(event)
            
            # Enforce per-container limits
            if len(container_events) > self.max_events_per_container:
                # Remove oldest events
                remove_count = len(container_events) - self.max_events_per_container
                removed = container_events[:remove_count]
                container_events[:remove_count] = []
                
                # Clean up correlation index
                for old_event in removed:
                    corr_events = self._events_by_correlation.get(old_event.correlation_id, [])
                    if old_event in corr_events:
                        corr_events.remove(old_event)
            
            # Store by correlation_id (for tracing)
            self._events_by_correlation[event.correlation_id].append(event)
            
    def read_events(
        self, 
        correlation_id: Optional[str] = None,
        source_container: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> Iterator[TracedEvent]:
        """Read events with container isolation awareness."""
        # Determine which events to search
        if correlation_id:
            # Correlation ID takes precedence (crosses container boundaries)
            events_to_search = self._events_by_correlation.get(correlation_id, [])
        elif source_container:
            # Container-specific search (respects isolation)
            events_to_search = self._events_by_container.get(source_container, [])
        else:
            # Search all events (be careful with this in production)
            events_to_search = []
            for container_events in self._events_by_container.values():
                events_to_search.extend(container_events)
                
        # Apply filters
        count = 0
        for event in events_to_search:
            if limit and count >= limit:
                break
                
            if event_type and event.event_type != event_type:
                continue
                
            if start_time and event.timestamp < start_time:
                continue
                
            if end_time and event.timestamp > end_time:
                continue
                
            if source_container and event.source_container != source_container:
                continue
                
            yield event
            count += 1
            
    def delete_events(self, correlation_id: str) -> int:
        """Delete events by correlation ID across all containers."""
        events = self._events_by_correlation.get(correlation_id, [])
        if not events:
            return 0
            
        # Remove from container storage
        for event in events:
            container_events = self._events_by_container.get(event.source_container, [])
            if event in container_events:
                container_events.remove(event)
                
        # Remove from correlation index
        del self._events_by_correlation[correlation_id]
        
        return len(events)
        
    def get_correlation_ids(
        self,
        source_container: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[str]:
        """Get unique correlation IDs with optional filtering."""
        correlation_ids = set()
        
        if source_container:
            # Search only within specific container (isolation)
            events = self._events_by_container.get(source_container, [])
        else:
            # Search across all containers
            events = []
            for container_events in self._events_by_container.values():
                events.extend(container_events)
                
        for event in events:
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
                
            correlation_ids.add(event.correlation_id)
            
        return sorted(list(correlation_ids))


class ParquetBackend:
    """Parquet file storage backend with container isolation."""
    
    def __init__(
        self, 
        base_path: str,
        partition_by: str = "daily",  # "hourly", "daily", "container"
        compression: str = "gzip"
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.partition_by = partition_by
        self.compression = compression
        self.logger = logging.getLogger(f"{__name__}.ParquetBackend")
        
        # Import here to make it optional
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            self.pa = pa
            self.pq = pq
        except ImportError:
            raise ImportError(
                "ParquetBackend requires pyarrow. Install with: pip install pyarrow"
            )
            
    def _get_partition_path(self, event: TracedEvent) -> Path:
        """Get partition path based on event and partitioning strategy."""
        if self.partition_by == "container":
            # Isolate by container
            return self.base_path / event.source_container / f"{event.timestamp.date()}.parquet"
        elif self.partition_by == "hourly":
            return self.base_path / f"{event.timestamp.strftime('%Y/%m/%d/%H')}.parquet"
        else:  # daily
            return self.base_path / f"{event.timestamp.strftime('%Y/%m/%d')}.parquet"
            
    def write_events(self, events: List[TracedEvent]) -> None:
        """Write events to Parquet files with partitioning."""
        if not events:
            return
            
        # Group events by partition
        partitions = defaultdict(list)
        for event in events:
            partition_path = self._get_partition_path(event)
            partitions[partition_path].append(event)
            
        # Write each partition
        for partition_path, partition_events in partitions.items():
            self._write_partition(partition_path, partition_events)
            
    def _write_partition(self, path: Path, events: List[TracedEvent]) -> None:
        """Write events to a specific partition file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert events to records
        records = []
        for event in events:
            record = {
                'event_id': event.event_id,
                'correlation_id': event.correlation_id,
                'causation_id': event.causation_id,
                'source_container': event.source_container,
                'target_container': event.target_container,
                'event_type': event.event_type,
                'timestamp': event.timestamp,
                'data': json.dumps(event.data),  # JSON serialize data
                'metadata': json.dumps(event.metadata) if event.metadata else None
            }
            records.append(record)
            
        # Create Arrow table
        table = self.pa.Table.from_pylist(records)
        
        # Write or append to Parquet file
        if path.exists():
            # Append to existing file
            existing_table = self.pq.read_table(path)
            combined_table = self.pa.concat_tables([existing_table, table])
            self.pq.write_table(
                combined_table, 
                path,
                compression=self.compression
            )
        else:
            # Write new file
            self.pq.write_table(
                table,
                path,
                compression=self.compression
            )
            
        self.logger.debug(f"Wrote {len(events)} events to {path}")
        
    def read_events(
        self, 
        correlation_id: Optional[str] = None,
        source_container: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> Iterator[TracedEvent]:
        """Read events from Parquet files with filtering."""
        # Find relevant partition files
        partition_files = self._find_partition_files(
            source_container, start_time, end_time
        )
        
        count = 0
        for partition_file in partition_files:
            if limit and count >= limit:
                break
                
            # Read partition with filters
            filters = []
            if correlation_id:
                filters.append(('correlation_id', '==', correlation_id))
            if source_container:
                filters.append(('source_container', '==', source_container))
            if event_type:
                filters.append(('event_type', '==', event_type))
                
            try:
                table = self.pq.read_table(
                    partition_file,
                    filters=filters if filters else None
                )
                
                # Convert to TracedEvents
                for row in table.to_pylist():
                    if limit and count >= limit:
                        break
                        
                    event = TracedEvent(
                        event_id=row['event_id'],
                        correlation_id=row['correlation_id'],
                        causation_id=row['causation_id'],
                        source_container=row['source_container'],
                        target_container=row['target_container'],
                        event_type=row['event_type'],
                        timestamp=row['timestamp'],
                        data=json.loads(row['data']),
                        metadata=json.loads(row['metadata']) if row['metadata'] else None
                    )
                    
                    # Apply time filters (if not handled by Parquet)
                    if start_time and event.timestamp < start_time:
                        continue
                    if end_time and event.timestamp > end_time:
                        continue
                        
                    yield event
                    count += 1
                    
            except Exception as e:
                self.logger.warning(f"Error reading partition {partition_file}: {e}")
                continue
                
    def _find_partition_files(
        self,
        source_container: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Path]:
        """Find partition files that may contain matching events."""
        partition_files = []
        
        if self.partition_by == "container" and source_container:
            # Container-based partitioning - respect isolation
            container_path = self.base_path / source_container
            if container_path.exists():
                partition_files.extend(container_path.glob("*.parquet"))
        else:
            # Time-based partitioning - may need to scan multiple files
            # This is a simplified implementation - could be optimized
            for parquet_file in self.base_path.rglob("*.parquet"):
                partition_files.append(parquet_file)
                
        return sorted(partition_files)
        
    def delete_events(self, correlation_id: str) -> int:
        """Delete events by correlation ID."""
        # This is expensive with Parquet - need to rewrite files
        # In production, consider marking as deleted instead
        deleted_count = 0
        
        for partition_file in self.base_path.rglob("*.parquet"):
            try:
                # Read partition
                table = self.pq.read_table(partition_file)
                df = table.to_pandas()
                
                # Filter out events with matching correlation_id
                original_count = len(df)
                df = df[df['correlation_id'] != correlation_id]
                filtered_count = len(df)
                
                if filtered_count < original_count:
                    # Rewrite file without deleted events
                    if filtered_count > 0:
                        new_table = self.pa.Table.from_pandas(df)
                        self.pq.write_table(
                            new_table,
                            partition_file,
                            compression=self.compression
                        )
                    else:
                        # Delete empty file
                        partition_file.unlink()
                        
                    deleted_count += original_count - filtered_count
                    
            except Exception as e:
                self.logger.warning(f"Error processing partition {partition_file}: {e}")
                
        return deleted_count
        
    def get_correlation_ids(
        self,
        source_container: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[str]:
        """Get unique correlation IDs."""
        correlation_ids = set()
        
        partition_files = self._find_partition_files(
            source_container, start_time, end_time
        )
        
        for partition_file in partition_files:
            try:
                # Read only correlation_id column for efficiency
                table = self.pq.read_table(
                    partition_file,
                    columns=['correlation_id', 'source_container', 'timestamp']
                )
                
                for row in table.to_pylist():
                    # Apply filters
                    if source_container and row['source_container'] != source_container:
                        continue
                    if start_time and row['timestamp'] < start_time:
                        continue
                    if end_time and row['timestamp'] > end_time:
                        continue
                        
                    correlation_ids.add(row['correlation_id'])
                    
            except Exception as e:
                self.logger.warning(f"Error reading partition {partition_file}: {e}")
                
        return sorted(list(correlation_ids))


# TimescaleDB backend would go here, following similar patterns
# but using PostgreSQL with TimescaleDB extension for time-series optimization