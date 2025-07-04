"""Parquet storage backend for data mining and analytics."""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

from ...core.events.protocols import EventStorageProtocol
from ...core.events.types import Event

logger = logging.getLogger(__name__)


class ParquetEventStorage(EventStorageProtocol):
    """
    Parquet storage backend for event traces - optimized for analytics.
    
    Designed for the two-layer data mining architecture:
    - SQL databases for high-level metrics and queries
    - Parquet files for detailed event analysis
    
    Features:
    - Columnar storage for efficient analytical queries
    - Correlation ID indexing for bridging metrics and events
    - Automatic compression and partitioning
    - Export functionality for data mining workflows
    """
    
    def __init__(self, base_path: Union[str, Path], 
                 correlation_id: str,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Parquet storage for event traces.
        
        Args:
            base_path: Base directory for parquet files
            correlation_id: Correlation ID for this trace session
            config: Configuration options
        """
        self.base_path = Path(base_path)
        self.correlation_id = correlation_id
        self.config = config or {}
        
        # Storage configuration
        self.compression = self.config.get('compression', 'snappy')
        self.partition_by_date = self.config.get('partition_by_date', True)
        self.batch_size = self.config.get('batch_size', 1000)
        
        # Event batching for efficient writes
        self._event_batch: List[Dict[str, Any]] = []
        self._events_written = 0
        
        # Ensure directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ParquetEventStorage initialized: {self.base_path} / {self.correlation_id}")
    
    def store(self, event: Event) -> None:
        """Store event in batch for efficient parquet writing."""
        # Convert event to dict with flattened structure for columnar storage
        event_dict = self._flatten_event(event)
        self._event_batch.append(event_dict)
        
        # Write batch when full
        if len(self._event_batch) >= self.batch_size:
            self._flush_batch()
    
    def _flatten_event(self, event: Event) -> Dict[str, Any]:
        """Flatten event structure for efficient columnar storage."""
        # Base event data
        flattened = {
            'correlation_id': event.correlation_id or self.correlation_id,
            'event_id': event.event_id,
            'event_type': event.event_type,
            'timestamp': event.timestamp,
            'source_id': event.source_id,
            'container_id': event.container_id,
            'causation_id': event.causation_id,
            'sequence_number': event.sequence_number,
        }
        
        # Flatten payload for columnar access
        if event.payload:
            for key, value in event.payload.items():
                # Prefix with payload_ to avoid column name conflicts
                flattened[f'payload_{key}'] = value
        
        # Flatten key metadata for analytics
        if event.metadata:
            # Trace info
            trace_info = event.metadata.get('trace_info', {})
            flattened.update({
                'tracer_id': trace_info.get('tracer_id'),
                'traced_at': trace_info.get('traced_at'),
                'trace_sequence': trace_info.get('trace_sequence'),
                'container_sequence': trace_info.get('container_sequence'),
            })
            
            # Timing info for latency analysis
            timing = event.metadata.get('timing', {})
            flattened.update({
                'event_created': timing.get('event_created'),
                'trace_enhanced': timing.get('trace_enhanced'),
            })
            
            # Container isolation info
            isolation = event.metadata.get('isolation', {})
            flattened.update({
                'source_container': isolation.get('source_container'),
                'container_trace_id': isolation.get('container_trace_id'),
                'isolated': isolation.get('isolated', False),
            })
        
        return flattened
    
    def _flush_batch(self) -> None:
        """Write current batch to parquet file."""
        if not self._event_batch:
            return
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(self._event_batch)
            
            # Determine file path
            file_path = self._get_batch_file_path()
            
            # Write with compression
            df.to_parquet(
                file_path,
                compression=self.compression,
                index=False
            )
            
            self._events_written += len(self._event_batch)
            logger.debug(f"Flushed {len(self._event_batch)} events to {file_path}")
            
            # Clear batch
            self._event_batch.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush event batch: {e}")
            raise
    
    def _get_batch_file_path(self) -> Path:
        """Get file path for current batch with optional date partitioning."""
        if self.partition_by_date:
            # Partition by date for efficient querying
            date_str = datetime.now().strftime('%Y-%m-%d')
            partition_dir = self.base_path / f"date={date_str}"
            partition_dir.mkdir(exist_ok=True)
            
            # Include timestamp to avoid conflicts
            timestamp = datetime.now().strftime('%H%M%S')
            filename = f"{self.correlation_id}_{timestamp}.parquet"
            return partition_dir / filename
        else:
            # Simple correlation-based naming
            batch_num = (self._events_written // self.batch_size) + 1
            filename = f"{self.correlation_id}_{batch_num:04d}.parquet"
            return self.base_path / filename
    
    def query(self, criteria: Dict[str, Any]) -> List[Event]:
        """Query events from stored parquet files."""
        # Load all parquet files for this correlation ID
        parquet_files = list(self.base_path.glob(f"**/*{self.correlation_id}*.parquet"))
        
        if not parquet_files:
            return []
        
        try:
            # Read all files into single DataFrame
            dfs = [pd.read_parquet(f) for f in parquet_files]
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Apply filtering criteria
            filtered_df = self._apply_query_filters(combined_df, criteria)
            
            # Convert back to Event objects
            events = [self._df_row_to_event(row) for _, row in filtered_df.iterrows()]
            
            return events
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def _apply_query_filters(self, df: pd.DataFrame, criteria: Dict[str, Any]) -> pd.DataFrame:
        """Apply query criteria to DataFrame."""
        filtered = df.copy()
        
        # Standard filters
        for key, value in criteria.items():
            if key in filtered.columns:
                if isinstance(value, list):
                    filtered = filtered[filtered[key].isin(value)]
                else:
                    filtered = filtered[filtered[key] == value]
        
        return filtered
    
    def _df_row_to_event(self, row: pd.Series) -> Event:
        """Convert DataFrame row back to Event object."""
        # Extract payload fields
        payload = {}
        for col in row.index:
            if col.startswith('payload_'):
                key = col[8:]  # Remove 'payload_' prefix
                payload[key] = row[col]
        
        # Reconstruct metadata
        metadata = {
            'trace_info': {
                'tracer_id': row.get('tracer_id'),
                'traced_at': row.get('traced_at'),
                'trace_sequence': row.get('trace_sequence'),
                'container_sequence': row.get('container_sequence'),
            },
            'timing': {
                'event_created': row.get('event_created'),
                'trace_enhanced': row.get('trace_enhanced'),
            },
            'isolation': {
                'source_container': row.get('source_container'),
                'container_trace_id': row.get('container_trace_id'),
                'isolated': row.get('isolated', False),
            }
        }
        
        # Create Event object
        return Event(
            event_id=row.get('event_id'),
            event_type=row.get('event_type'),
            timestamp=row.get('timestamp'),
            source_id=row.get('source_id'),
            container_id=row.get('container_id'),
            correlation_id=row.get('correlation_id'),
            causation_id=row.get('causation_id'),
            sequence_number=row.get('sequence_number'),
            payload=payload,
            metadata=metadata
        )
    
    def count(self) -> int:
        """Count total events stored."""
        return self._events_written + len(self._event_batch)
    
    def clear(self) -> None:
        """Clear all stored events."""
        # Flush any pending batch first
        if self._event_batch:
            self._flush_batch()
        
        # Remove all parquet files for this correlation ID
        parquet_files = list(self.base_path.glob(f"**/*{self.correlation_id}*.parquet"))
        for file_path in parquet_files:
            file_path.unlink()
        
        self._events_written = 0
        logger.info(f"Cleared all events for correlation {self.correlation_id}")
    
    def export_to_file(self, filepath: str) -> None:
        """Export all events to a single parquet file for analysis."""
        # Flush pending batch
        if self._event_batch:
            self._flush_batch()
        
        # Load all parquet files
        parquet_files = list(self.base_path.glob(f"**/*{self.correlation_id}*.parquet"))
        
        if not parquet_files:
            logger.warning(f"No parquet files found for correlation {self.correlation_id}")
            return
        
        try:
            # Combine all files
            dfs = [pd.read_parquet(f) for f in parquet_files]
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Sort by timestamp for proper event ordering
            combined_df = combined_df.sort_values('timestamp')
            
            # Export to single file
            combined_df.to_parquet(filepath, compression=self.compression, index=False)
            
            logger.info(f"Exported {len(combined_df)} events to {filepath}")
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get summary statistics for data mining analysis."""
        # Flush pending batch for accurate count
        if self._event_batch:
            self._flush_batch()
        
        # Get file statistics
        parquet_files = list(self.base_path.glob(f"**/*{self.correlation_id}*.parquet"))
        
        total_size_mb = sum(f.stat().st_size for f in parquet_files) / (1024 * 1024)
        
        return {
            'correlation_id': self.correlation_id,
            'total_events': self._events_written,
            'parquet_files': len(parquet_files),
            'total_size_mb': round(total_size_mb, 2),
            'compression': self.compression,
            'partitioned_by_date': self.partition_by_date,
            'storage_path': str(self.base_path),
            'ready_for_data_mining': True
        }


def create_parquet_storage(base_path: Union[str, Path], 
                          correlation_id: str,
                          config: Optional[Dict[str, Any]] = None) -> ParquetEventStorage:
    """
    Factory function to create Parquet storage backend.
    
    Args:
        base_path: Base directory for parquet files
        correlation_id: Correlation ID for this trace session
        config: Storage configuration options
    
    Returns:
        Configured ParquetEventStorage instance
    """
    return ParquetEventStorage(base_path, correlation_id, config)
