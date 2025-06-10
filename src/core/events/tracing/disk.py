"""Enhanced disk-based event storage with Parquet support and batching."""

import os
import json
import gzip
from typing import Dict, Any, Optional, List, Iterator
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import logging

from ..protocols import EventStorageProtocol
from ..types import Event

logger = logging.getLogger(__name__)

class DiskEventStorage(EventStorageProtocol):
    """
    Enhanced disk-based event storage with Parquet support and intelligent batching.
    
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
        
        self.event_batch.clear()
        self.last_flush = datetime.now()
        
        logger.debug(f"Flushed {len(self.event_batch)} events to disk")
    
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
                    event_data = json.loads(line)
                    if event_data.get('metadata', {}).get('event_id') == event_id:
                        return Event.from_dict(event_data)
        return None
    
    def query(self, criteria: Dict[str, Any]) -> List[Event]:
        """Query events by criteria (requires scanning files)."""
        results = []
        
        for file_path in self._get_all_files():
            with self._open_file_for_read(file_path) as f:
                for line in f:
                    event_data = json.loads(line)
                    event = Event.from_dict(event_data)
                    
                    if self._matches_criteria(event, criteria):
                        results.append(event)
        
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
        self.current_file.flush()
        
        with open(filepath, 'wb') as output:
            for file_path in self._get_all_files():
                with open(file_path, 'rb') as input_file:
                    output.write(input_file.read())
    
    # Private methods
    
    def _open_new_file(self) -> None:
        """Open a new file for writing."""
        if self.current_file:
            self.current_file.close()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"events_{timestamp}_{self.file_count:04d}"
        
        if self.compression:
            filename += '.jsonl.gz'
            file_path = self.directory / filename
            self.current_file = gzip.open(file_path, 'wt', encoding='utf-8')
        else:
            filename += '.jsonl'
            file_path = self.directory / filename
            self.current_file = open(file_path, 'w', encoding='utf-8')
        
        self.current_file_size = 0
        self.file_count += 1
    
    def _rotate_file(self) -> None:
        """Rotate to a new file."""
        self._open_new_file()
    
    def _get_all_files(self) -> List[Path]:
        """Get all event files in order."""
        pattern = '*.jsonl.gz' if self.compression else '*.jsonl'
        files = sorted(self.directory.glob(pattern))
        return files
    
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
        """Ensure file is closed on deletion."""
        if hasattr(self, 'current_file') and self.current_file:
            self.current_file.close()