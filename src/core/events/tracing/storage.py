"""Event storage backends - memory and disk.
Consolidated from storage/ subdirectory.
"""

from typing import Dict, Any, Optional, List, Set
from collections import deque, defaultdict
from datetime import datetime
import json
import os
import gzip
from pathlib import Path
import logging

from ..protocols import EventStorageProtocol
from ..types import Event

logger = logging.getLogger(__name__)


class MemoryEventStorage(EventStorageProtocol):
    """
    In-memory event storage with configurable retention.
    
    Used by EventTracer for temporary storage during execution.
    Supports multiple indices for efficient querying.
    """
    
    def __init__(self, max_size: Optional[int] = None, 
                 enable_indices: bool = True,
                 retention_policy: str = "sliding_window"):
        """
        Initialize memory storage.
        
        Args:
            max_size: Maximum events to store (None for unlimited)
            enable_indices: Whether to maintain indices for fast queries
            retention_policy: How to handle event retention
                - "sliding_window": Keep last N events
                - "trade_complete": Keep events until trade completes
                - "none": No automatic pruning
        """
        self.max_size = max_size
        self.enable_indices = enable_indices
        self.retention_policy = retention_policy
        
        # Primary storage
        self.events: deque = deque(maxlen=max_size) if max_size else deque()
        
        # Indices for fast lookup with size limits
        if enable_indices:
            self._event_id_index: Dict[str, Event] = {}
            self._correlation_index: Dict[str, List[Event]] = defaultdict(list)
            self._type_index: Dict[str, List[Event]] = defaultdict(list)
            # Track index sizes to prevent unbounded growth
            self._max_index_size = max_size * 2 if max_size else 100000
        
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
        
        # Apply retention policy
        self.apply_retention_policy(event)
    
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
        """Prune events matching criteria - efficient bulk operation."""
        # Create new deque to avoid O(n) removals
        new_events = deque(maxlen=self.max_size)
        indices_to_rebuild = False
        pruned = 0
        
        for event in self.events:
            if not self._matches_criteria(event, criteria):
                new_events.append(event)
            else:
                pruned += 1
                indices_to_rebuild = True
        
        if indices_to_rebuild:
            # Rebuild indices from scratch - more efficient for bulk operations
            self.events = new_events
            self._rebuild_indices()
        
        return pruned
    
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
        """Add event to indices with size management."""
        # Check if indices are getting too large
        if self._should_cleanup_indices():
            self._cleanup_old_indices()
        
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
            
            # Handle special criteria
            if key == 'timestamp_before':
                if hasattr(event, 'timestamp') and event.timestamp:
                    if event.timestamp >= value:
                        return False
                continue
            
            if key == 'exclude_types':
                if event.event_type in value:
                    return False
                continue
                
            # Standard attribute matching
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
    
    def apply_retention_policy(self, event: Event) -> None:
        """Apply configured retention policy after storing event."""
        if self.retention_policy == "trade_complete":
            # Check if this event closes a trade
            if event.event_type == "POSITION_CLOSE" and event.correlation_id:
                # Prune all events for this trade
                self.prune({
                    'correlation_id': event.correlation_id
                })
        elif self.retention_policy == "sliding_window":
            # Automatic with deque maxlen
            pass
        elif self.retention_policy == "minimal":
            # Only keep essential events
            if event.event_type not in ["PORTFOLIO_UPDATE", "POSITION_OPEN", "POSITION_CLOSE"]:
                # Remove non-essential events older than 1 minute
                from datetime import timedelta
                cutoff_time = datetime.now() - timedelta(minutes=1)
                self.prune({
                    'timestamp_before': cutoff_time,
                    'exclude_types': ["PORTFOLIO_UPDATE", "POSITION_OPEN", "POSITION_CLOSE"]
                })
    
    def _rebuild_indices(self) -> None:
        """Rebuild all indices from current events."""
        if not self.enable_indices:
            return
            
        # Clear existing indices
        self._event_id_index.clear()
        self._correlation_index.clear()
        self._type_index.clear()
        
        # Rebuild from events
        for event in self.events:
            self._add_to_indices(event)
    
    def _should_cleanup_indices(self) -> bool:
        """Check if indices need cleanup."""
        if not self.enable_indices:
            return False
            
        total_index_entries = (
            len(self._event_id_index) +
            sum(len(events) for events in self._correlation_index.values()) +
            sum(len(events) for events in self._type_index.values())
        )
        return total_index_entries > self._max_index_size
    
    def _cleanup_old_indices(self) -> None:
        """Remove index entries for events no longer in main storage."""
        if not self.enable_indices:
            return
            
        current_event_ids = {e.metadata.get('event_id') for e in self.events if e.metadata.get('event_id')}
        
        # Clean event ID index
        self._event_id_index = {
            k: v for k, v in self._event_id_index.items() 
            if k in current_event_ids
        }
        
        # Clean correlation index
        for corr_id in list(self._correlation_index.keys()):
            self._correlation_index[corr_id] = [
                e for e in self._correlation_index[corr_id] 
                if e in self.events
            ]
            if not self._correlation_index[corr_id]:
                del self._correlation_index[corr_id]
        
        # Clean type index
        for event_type in list(self._type_index.keys()):
            self._type_index[event_type] = [
                e for e in self._type_index[event_type]
                if e in self.events
            ]
            if not self._type_index[event_type]:
                del self._type_index[event_type]


class DiskEventStorage(EventStorageProtocol):
    """
    Enhanced disk-based event storage with Parquet support and batching.
    
    Features:
    - Parquet format for efficient columnar storage and querying
    - Intelligent batching with flush policies
    - Container-based partitioning for isolation
    - Compression and file rotation
    - Fast querying with predicate pushdown
    """
    
    def __init__(
        self,
        directory: str = './traces',
        format: str = 'parquet',  # 'parquet' or 'jsonl'
        batch_size: int = 1000,
        max_file_size_mb: int = 100,
        compression: str = 'snappy',  # For parquet: snappy, gzip, lz4
        partition_by: str = 'container',  # 'container', 'daily', 'hourly'
        container_isolation: bool = True
    ):
        """
        Initialize enhanced disk storage.
        
        Args:
            directory: Directory to store event files
            format: Storage format ('parquet' or 'jsonl')
            batch_size: Events to batch before writing
            max_file_size_mb: Maximum size per file before rotation
            compression: Compression algorithm
            partition_by: Partitioning strategy
            container_isolation: Whether to isolate events by container
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        
        self.format = format
        self.batch_size = batch_size
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.compression = compression
        self.partition_by = partition_by
        self.container_isolation = container_isolation
        
        # Batching state
        self.event_batch: List[Event] = []
        self.last_flush = datetime.now()
        self.flush_interval = 5.0  # seconds
        
        # File management
        self.current_files: Dict[str, Any] = {}  # partition -> file handle
        self.file_sizes: Dict[str, int] = {}
        self.file_counts: Dict[str, int] = defaultdict(int)
        self._event_count = 0
        
        # Parquet support
        self.parquet_available = False
        if format == 'parquet':
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                self.pa = pa
                self.pq = pq
                self.parquet_available = True
                logger.info("Parquet support enabled")
            except ImportError:
                logger.warning("Parquet not available, falling back to JSONL format")
                self.format = 'jsonl'
    
    def store(self, event: Event) -> None:
        """Store event with intelligent batching."""
        self.event_batch.append(event)
        self._event_count += 1
        
        # Check if we should flush
        should_flush = (
            len(self.event_batch) >= self.batch_size or
            (datetime.now() - self.last_flush).total_seconds() > self.flush_interval
        )
        
        if should_flush:
            self.flush()
    
    def flush(self) -> None:
        """Flush pending events to storage."""
        if not self.event_batch:
            return
        
        if self.format == 'parquet' and self.parquet_available:
            self._flush_parquet()
        else:
            self._flush_jsonl()
        
        batch_size = len(self.event_batch)
        self.event_batch.clear()
        self.last_flush = datetime.now()
        
        logger.debug(f"Flushed {batch_size} events to disk")
    
    def _flush_parquet(self) -> None:
        """Flush events to Parquet format with partitioning."""
        # Group events by partition
        partitions = defaultdict(list)
        
        for event in self.event_batch:
            partition_key = self._get_partition_key(event)
            partitions[partition_key].append(event)
        
        # Write each partition
        for partition_key, events in partitions.items():
            self._write_parquet_partition(partition_key, events)
    
    def _flush_jsonl(self) -> None:
        """Flush events to JSONL format."""
        # Group by partition if container isolation enabled
        if self.container_isolation:
            partitions = defaultdict(list)
            for event in self.event_batch:
                partition_key = self._get_partition_key(event)
                partitions[partition_key].append(event)
        else:
            partitions = {'default': self.event_batch}
        
        # Write each partition
        for partition_key, events in partitions.items():
            self._write_jsonl_partition(partition_key, events)
    
    def _get_partition_key(self, event: Event) -> str:
        """Get partition key for event."""
        if self.partition_by == 'container' and event.container_id:
            return f"container_{event.container_id}"
        elif self.partition_by == 'daily':
            return event.timestamp.strftime('%Y-%m-%d')
        elif self.partition_by == 'hourly':
            return event.timestamp.strftime('%Y-%m-%d_%H')
        else:
            return 'default'
    
    def _write_parquet_partition(self, partition_key: str, events: List[Event]) -> None:
        """Write events to Parquet partition."""
        # Convert events to records
        records = []
        for event in events:
            record = {
                'event_id': event.event_id,
                'event_type': event.event_type,
                'source_id': event.source_id,
                'container_id': event.container_id,
                'correlation_id': event.correlation_id,
                'causation_id': event.causation_id,
                'target_container': event.target_container,
                'sequence_number': event.sequence_number,
                'timestamp': event.timestamp,
                'payload': json.dumps(event.payload),
                'metadata': json.dumps(event.metadata)
            }
            records.append(record)
        
        # Create Arrow table
        table = self.pa.Table.from_pylist(records)
        
        # Get partition path
        partition_path = self._get_partition_path(partition_key, 'parquet')
        partition_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write or append
        if partition_path.exists():
            # Append to existing file
            existing_table = self.pq.read_table(partition_path)
            combined_table = self.pa.concat_tables([existing_table, table])
            self.pq.write_table(combined_table, partition_path, compression=self.compression)
        else:
            # Write new file
            self.pq.write_table(table, partition_path, compression=self.compression)
    
    def _write_jsonl_partition(self, partition_key: str, events: List[Event]) -> None:
        """Write events to JSONL partition."""
        partition_path = self._get_partition_path(partition_key, 'jsonl')
        partition_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open file for append
        if self.compression == 'gzip':
            with gzip.open(f"{partition_path}.gz", 'at', encoding='utf-8') as f:
                for event in events:
                    f.write(json.dumps(event.to_dict(), default=str) + '\n')
        else:
            with open(partition_path, 'a', encoding='utf-8') as f:
                for event in events:
                    f.write(json.dumps(event.to_dict(), default=str) + '\n')
    
    def _get_partition_path(self, partition_key: str, format: str) -> Path:
        """Get file path for partition."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{partition_key}_{timestamp}.{format}"
        return self.directory / partition_key / filename
    
    def retrieve(self, event_id: str) -> Optional[Event]:
        """Retrieve event by ID (requires scanning files)."""
        for file_path in self._get_all_files():
            with self._open_file_for_read(file_path) as f:
                for line in f:
                    try:
                        event_data = json.loads(line)
                        if event_data.get('metadata', {}).get('event_id') == event_id:
                            return Event.from_dict(event_data)
                    except json.JSONDecodeError:
                        continue
        return None
    
    def query(self, criteria: Dict[str, Any]) -> List[Event]:
        """Query events by criteria (requires scanning files)."""
        results = []
        
        for file_path in self._get_all_files():
            with self._open_file_for_read(file_path) as f:
                for line in f:
                    try:
                        event_data = json.loads(line)
                        event = Event.from_dict(event_data)
                        
                        if self._matches_criteria(event, criteria):
                            results.append(event)
                    except (json.JSONDecodeError, Exception):
                        continue
        
        return results
    
    def prune(self, criteria: Dict[str, Any]) -> int:
        """Pruning not supported for disk storage."""
        # Disk storage doesn't support pruning individual events
        # Would need to rewrite files
        return 0
    
    def count(self) -> int:
        """Get total event count."""
        return self._event_count
    
    def export_to_file(self, filepath: str) -> None:
        """Export all events to a single file."""
        self.flush()  # Ensure all events are written
        
        with open(filepath, 'w') as output:
            for file_path in self._get_all_files():
                with self._open_file_for_read(file_path) as input_file:
                    for line in input_file:
                        output.write(line)
    
    # Private methods
    
    def _get_all_files(self) -> List[Path]:
        """Get all event files in order."""
        patterns = ['*.jsonl.gz', '*.jsonl', '*.parquet'] if self.compression else ['*.jsonl', '*.parquet']
        files = []
        for pattern in patterns:
            files.extend(self.directory.rglob(pattern))
        return sorted(files)
    
    def _open_file_for_read(self, file_path: Path):
        """Open file for reading."""
        if file_path.suffix == '.gz':
            return gzip.open(file_path, 'rt', encoding='utf-8')
        else:
            return open(file_path, 'r', encoding='utf-8')
    
    def _matches_criteria(self, event: Event, criteria: Dict[str, Any]) -> bool:
        """Check if event matches criteria."""
        for key, value in criteria.items():
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
    
    def __del__(self):
        """Ensure files are flushed on deletion."""
        try:
            self.flush()
        except:
            pass


def create_storage_backend(backend_type: str, config: Dict[str, Any]) -> EventStorageProtocol:
    """
    Create storage backend from configuration.
    
    Args:
        backend_type: 'memory', 'disk', 'hierarchical', or 'analytics_parquet'
        config: Backend-specific configuration
        
    Returns:
        Storage backend instance
    """
    if backend_type == 'memory':
        return MemoryEventStorage(
            max_size=config.get('max_size'),
            enable_indices=config.get('enable_indices', True)
        )
    elif backend_type == 'disk':
        return DiskEventStorage(
            directory=config.get('directory', './traces'),
            format=config.get('format', 'jsonl'),
            batch_size=config.get('batch_size', 1000),
            max_file_size_mb=config.get('max_file_size_mb', 100),
            compression=config.get('compression', 'gzip'),
            partition_by=config.get('partition_by', 'container'),
            container_isolation=config.get('container_isolation', True)
        )
    elif backend_type == 'hierarchical':
        # Import here to avoid circular dependency
        from ..storage.hierarchical import HierarchicalEventStorage, HierarchicalStorageConfig
        
        # Create configuration object
        storage_config = HierarchicalStorageConfig(
            base_dir=config.get('base_dir', './workspaces'),
            format=config.get('format', 'parquet'),
            compression=config.get('compression', 'snappy'),
            batch_size=config.get('batch_size', 1000),  # This will get the batch_size from config
            max_memory_mb=config.get('max_memory_mb', 100),
            enable_indices=config.get('enable_indices', True),
            retention_days=config.get('retention_days'),
            archive_completed_workflows=config.get('archive_completed_workflows', True)
        )
        
        storage = HierarchicalEventStorage(storage_config)
        
        # Set context if provided
        workflow_id = config.get('workflow_id')
        phase_name = config.get('phase_name')
        container_id = config.get('container_id')
        
        if workflow_id:
            storage.set_context(
                workflow_id=workflow_id,
                phase_name=phase_name,
                container_id=container_id
            )
        
        return storage
    elif backend_type == 'analytics_parquet':
        # Import here to avoid circular dependency
        from .storage.parquet_backend import ParquetEventStorage
        
        # Extract required parameters
        correlation_id = config.get('correlation_id', 'default_trace')
        base_path = config.get('base_path', './analytics/traces')
        
        return ParquetEventStorage(
            base_path=base_path,
            correlation_id=correlation_id,
            config=config
        )
    else:
        raise ValueError(f"Unknown storage backend: {backend_type}")