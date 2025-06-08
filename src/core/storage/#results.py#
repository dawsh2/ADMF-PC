"""
Hybrid results storage for ADMF-PC.

Uses different storage backends optimized for different data types:
- Parquet: For analytical data (metrics, trades) - columnar, compressed
- JSONL: For event streams - append-friendly, streamable
- SQLite: For metadata and indexing - queryable, ACID
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import json
import gzip
import sqlite3
import logging

logger = logging.getLogger(__name__)


@dataclass
class HybridResultStore:
    """
    Manages different storage backends for different data types.
    
    This provides efficient storage with appropriate retention policies
    for different types of trading system data.
    """
    
    base_path: Path
    
    # Different retention periods for different data types
    retention_policies: Dict[str, Optional[timedelta]] = field(default_factory=lambda: {
        'metrics': timedelta(days=365),      # Keep for analysis
        'trades': timedelta(days=90),        # Compliance requirements
        'events': timedelta(days=7),         # Large, short-term debugging
        'summary': None                      # Keep forever
    })
    
    def __post_init__(self):
        """Create directory structure."""
        self.base_path = Path(self.base_path)
        self.metrics_dir = self.base_path / 'metrics'
        self.trades_dir = self.base_path / 'trades'
        self.events_dir = self.base_path / 'events'
        self.metadata_db = self.base_path / 'metadata.db'
        
        # Create directories
        for dir_path in [self.metrics_dir, self.trades_dir, self.events_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata database
        self._init_metadata_db()
    
    def _init_metadata_db(self):
        """Initialize SQLite database for metadata."""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    workflow_type TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    metadata JSON
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS phases (
                    phase_id TEXT PRIMARY KEY,
                    workflow_id TEXT,
                    phase_name TEXT,
                    phase_type TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    status TEXT,
                    metadata JSON,
                    FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS containers (
                    container_id TEXT PRIMARY KEY,
                    phase_id TEXT,
                    container_type TEXT,
                    created_at TIMESTAMP,
                    final_metrics JSON,
                    FOREIGN KEY (phase_id) REFERENCES phases(phase_id)
                )
            """)
    
    def store_metrics(self, metrics: pd.DataFrame, phase_id: str, container_id: str) -> None:
        """
        Store metrics in Parquet for efficient analysis.
        
        Parquet provides:
        - Columnar storage for efficient queries
        - Excellent compression
        - Schema preservation
        """
        # Create subdirectory for this phase
        phase_dir = self.metrics_dir / phase_id
        phase_dir.mkdir(exist_ok=True)
        
        # Store with container ID
        path = phase_dir / f'{container_id}_metrics.parquet'
        metrics.to_parquet(path, compression='snappy')
        
        logger.info(f"Stored {len(metrics)} metric rows for {container_id}")
    
    def store_trades(self, trades: pd.DataFrame, phase_id: str, container_id: str) -> None:
        """Store trade records in Parquet format."""
        phase_dir = self.trades_dir / phase_id
        phase_dir.mkdir(exist_ok=True)
        
        path = phase_dir / f'{container_id}_trades.parquet'
        trades.to_parquet(path, compression='snappy')
        
        logger.info(f"Stored {len(trades)} trades for {container_id}")
    
    def store_events(self, events: List[Dict[str, Any]], phase_id: str, 
                    container_id: str, event_type: str) -> None:
        """
        Store events in compressed JSONL format.
        
        JSONL is ideal for:
        - Append operations during execution
        - Streaming reads for analysis
        - Human-readable debugging
        """
        phase_dir = self.events_dir / phase_id
        phase_dir.mkdir(exist_ok=True)
        
        # Group by event type for easier filtering
        path = phase_dir / f'{container_id}_{event_type}.jsonl.gz'
        
        # Append mode for streaming writes
        with gzip.open(path, 'at', encoding='utf-8') as f:
            for event in events:
                f.write(json.dumps(event, default=str) + '\n')
        
        logger.debug(f"Stored {len(events)} {event_type} events for {container_id}")
    
    def store_workflow_metadata(self, workflow_id: str, metadata: Dict[str, Any]) -> None:
        """Store workflow metadata in SQLite."""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO workflows 
                (workflow_id, workflow_type, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                workflow_id,
                metadata.get('workflow_type', 'unknown'),
                metadata.get('created_at', datetime.now()),
                datetime.now(),
                json.dumps(metadata)
            ))
    
    def store_phase_metadata(self, phase_id: str, workflow_id: str, 
                           phase_name: str, metadata: Dict[str, Any]) -> None:
        """Store phase metadata in SQLite."""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO phases
                (phase_id, workflow_id, phase_name, phase_type, started_at, 
                 completed_at, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                phase_id,
                workflow_id,
                phase_name,
                metadata.get('phase_type', 'unknown'),
                metadata.get('started_at', datetime.now()),
                metadata.get('completed_at'),
                metadata.get('status', 'running'),
                json.dumps(metadata)
            ))
    
    def store_final_results(self, container_id: str, phase_id: str, 
                          results: Dict[str, Any]) -> None:
        """Store final container results."""
        # Store metrics summary in SQLite
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO containers
                (container_id, phase_id, container_type, created_at, final_metrics)
                VALUES (?, ?, ?, ?, ?)
            """, (
                container_id,
                phase_id,
                results.get('container_type', 'unknown'),
                datetime.now(),
                json.dumps(results.get('metrics', {}))
            ))
        
        # Store detailed results as JSON
        summary_path = self.base_path / 'summaries' / f'{phase_id}'
        summary_path.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path / f'{container_id}_summary.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def query_metrics(self, phase_id: str, metric_names: List[str], 
                     container_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Efficiently query specific metrics.
        
        Parquet allows column selection without loading full file.
        """
        phase_dir = self.metrics_dir / phase_id
        if not phase_dir.exists():
            return pd.DataFrame()
        
        # Find relevant metric files
        pattern = f'*_metrics.parquet' if not container_filter else f'{container_filter}_metrics.parquet'
        metric_files = list(phase_dir.glob(pattern))
        
        if not metric_files:
            return pd.DataFrame()
        
        # Load and concatenate
        dfs = []
        for file_path in metric_files:
            # Parquet allows efficient column selection
            df = pd.read_parquet(file_path, columns=metric_names)
            df['container_id'] = file_path.stem.replace('_metrics', '')
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)
    
    def load_events(self, phase_id: str, container_id: str, 
                   event_type: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load events from compressed JSONL."""
        path = self.events_dir / phase_id / f'{container_id}_{event_type}.jsonl.gz'
        
        if not path.exists():
            return []
        
        events = []
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                events.append(json.loads(line))
        
        return events
    
    def query_workflows(self, workflow_type: Optional[str] = None, 
                       start_date: Optional[datetime] = None) -> pd.DataFrame:
        """Query workflow metadata."""
        query = "SELECT * FROM workflows WHERE 1=1"
        params = []
        
        if workflow_type:
            query += " AND workflow_type = ?"
            params.append(workflow_type)
        
        if start_date:
            query += " AND created_at >= ?"
            params.append(start_date)
        
        with sqlite3.connect(self.metadata_db) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            # Parse JSON metadata
            df['metadata'] = df['metadata'].apply(json.loads)
        
        return df
    
    def cleanup_old_data(self) -> None:
        """Remove data older than retention policy."""
        now = datetime.now()
        
        for data_type, retention in self.retention_policies.items():
            if retention is None:
                continue  # Keep forever
            
            cutoff_date = now - retention
            
            if data_type == 'events':
                # Clean up old event files
                for event_file in self.events_dir.rglob('*.jsonl.gz'):
                    if datetime.fromtimestamp(event_file.stat().st_mtime) < cutoff_date:
                        event_file.unlink()
                        logger.info(f"Deleted old event file: {event_file}")
            
            elif data_type == 'metrics':
                # Clean up old metric files
                for metric_file in self.metrics_dir.rglob('*.parquet'):
                    if datetime.fromtimestamp(metric_file.stat().st_mtime) < cutoff_date:
                        metric_file.unlink()
                        logger.info(f"Deleted old metric file: {metric_file}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        def get_dir_size(path: Path) -> int:
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        
        return {
            'metrics_size_mb': get_dir_size(self.metrics_dir) / 1024 / 1024,
            'trades_size_mb': get_dir_size(self.trades_dir) / 1024 / 1024,
            'events_size_mb': get_dir_size(self.events_dir) / 1024 / 1024,
            'total_size_mb': get_dir_size(self.base_path) / 1024 / 1024,
            'workflow_count': self._count_workflows(),
            'retention_policies': {k: str(v) for k, v in self.retention_policies.items()}
        }
    
    def _count_workflows(self) -> int:
        """Count total workflows in database."""
        with sqlite3.connect(self.metadata_db) as conn:
            result = conn.execute("SELECT COUNT(*) FROM workflows").fetchone()
            return result[0] if result else 0


@dataclass
class ResultCollector:
    """
    Collects results during execution and stores them efficiently.
    
    This component can be added to containers to automatically collect
    and store results using the hybrid storage approach.
    """
    
    store: HybridResultStore
    phase_id: str
    container_id: str
    
    # Buffers for batch writing
    metric_buffer: List[Dict[str, Any]] = field(default_factory=list)
    trade_buffer: List[Dict[str, Any]] = field(default_factory=list) 
    event_buffer: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    
    # Buffer sizes
    metric_buffer_size: int = 1000
    trade_buffer_size: int = 100
    event_buffer_size: int = 100
    
    def collect_metric(self, timestamp: datetime, metric_name: str, value: float,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Collect a metric data point."""
        metric = {
            'timestamp': timestamp,
            'metric_name': metric_name,
            'value': value
        }
        if metadata:
            metric.update(metadata)
        
        self.metric_buffer.append(metric)
        
        # Flush if buffer full
        if len(self.metric_buffer) >= self.metric_buffer_size:
            self.flush_metrics()
    
    def collect_trade(self, trade: Dict[str, Any]) -> None:
        """Collect a trade record."""
        self.trade_buffer.append(trade)
        
        if len(self.trade_buffer) >= self.trade_buffer_size:
            self.flush_trades()
    
    def collect_event(self, event_type: str, event: Dict[str, Any]) -> None:
        """Collect an event for debugging."""
        self.event_buffer[event_type].append(event)
        
        if len(self.event_buffer[event_type]) >= self.event_buffer_size:
            self.flush_events(event_type)
    
    def flush_metrics(self) -> None:
        """Flush metric buffer to storage."""
        if self.metric_buffer:
            df = pd.DataFrame(self.metric_buffer)
            self.store.store_metrics(df, self.phase_id, self.container_id)
            self.metric_buffer.clear()
    
    def flush_trades(self) -> None:
        """Flush trade buffer to storage."""
        if self.trade_buffer:
            df = pd.DataFrame(self.trade_buffer)
            self.store.store_trades(df, self.phase_id, self.container_id)
            self.trade_buffer.clear()
    
    def flush_events(self, event_type: Optional[str] = None) -> None:
        """Flush event buffer to storage."""
        if event_type:
            # Flush specific event type
            if event_type in self.event_buffer and self.event_buffer[event_type]:
                self.store.store_events(
                    self.event_buffer[event_type],
                    self.phase_id,
                    self.container_id,
                    event_type
                )
                self.event_buffer[event_type].clear()
        else:
            # Flush all event types
            for evt_type, events in self.event_buffer.items():
                if events:
                    self.store.store_events(events, self.phase_id, self.container_id, evt_type)
            self.event_buffer.clear()
    
    def flush_all(self) -> None:
        """Flush all buffers."""
        self.flush_metrics()
        self.flush_trades()
        self.flush_events()
    
    def finalize(self, results: Dict[str, Any]) -> None:
        """Finalize and store results."""
        # Flush any remaining data
        self.flush_all()
        
        # Store final results
        self.store.store_final_results(self.container_id, self.phase_id, results)