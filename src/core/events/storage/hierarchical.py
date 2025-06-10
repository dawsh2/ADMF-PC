"""
Hierarchical Event Storage for Parallelized Backtests

This implements the vision from signal-storage-replay.md and data-mining-architecture.md:
- Per-container event traces with correlation IDs
- Hierarchical directory structure for parallel portfolio analysis
- Sparse storage focusing on meaningful events (signals, trades, regime changes)
- Parquet format for efficient columnar storage and querying
"""

from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import json
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import dataclass, field

from ..protocols import EventStorageProtocol
from ..types import Event, EventType

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalStorageConfig:
    """Configuration for hierarchical event storage."""
    
    # Base directory for all results
    base_dir: str = "./results"
    
    # Storage format and options
    format: str = "parquet"  # parquet or jsonl
    compression: str = "snappy"  # snappy, gzip, lz4
    
    # Sparse storage settings
    sparse_event_types: Set[str] = field(default_factory=lambda: {
        EventType.SIGNAL.value,
        EventType.POSITION_OPEN.value,
        EventType.POSITION_CLOSE.value,
        EventType.FILL.value,
        EventType.REGIME_CHANGE.value,
        EventType.CLASSIFIER_CHANGE.value,
        EventType.RISK_BREACH.value
    })
    
    # Hierarchical structure
    enable_container_isolation: bool = True
    enable_workflow_grouping: bool = True
    enable_phase_grouping: bool = True
    
    # Performance settings
    batch_size: int = 1000
    max_memory_mb: int = 100
    enable_indices: bool = True
    
    # Retention settings
    retention_days: Optional[int] = None
    archive_completed_workflows: bool = True


class HierarchicalEventStorage(EventStorageProtocol):
    """
    Hierarchical storage optimized for parallel portfolio analysis.
    
    Directory structure:
    results/
    └── {workflow_id}_{timestamp}/
        ├── metadata.json              # Workflow metadata
        ├── phases/
        │   ├── signal_generation/
        │   │   ├── feature_container/
        │   │   │   ├── events.parquet
        │   │   │   └── metrics.json
        │   │   └── summary.json
        │   └── backtest/
        │       ├── portfolio_c0001/   # Per-portfolio traces
        │       │   ├── events.parquet
        │       │   ├── signals.parquet
        │       │   ├── trades.parquet
        │       │   └── metrics.json
        │       ├── portfolio_c0002/
        │       │   └── ...
        │       ├── execution/
        │       │   └── events.parquet
        │       └── summary.json
        └── analysis/
            ├── pattern_library.parquet
            └── optimization_results.json
    """
    
    def __init__(self, config: HierarchicalStorageConfig):
        """Initialize hierarchical storage."""
        self.config = config
        self.base_dir = Path(config.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Current context
        self.workflow_id: Optional[str] = None
        self.phase_name: Optional[str] = None
        self.container_id: Optional[str] = None
        
        # In-memory buffers per container
        self.event_buffers: Dict[str, List[Event]] = defaultdict(list)
        self.buffer_sizes: Dict[str, int] = defaultdict(int)
        
        # Sparse indices for fast lookup
        self.signal_indices: Dict[str, List[Tuple[int, str]]] = defaultdict(list)  # container -> [(idx, event_id)]
        self.trade_indices: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        self.regime_changes: List[Dict[str, Any]] = []
        
        # Statistics
        self.event_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._total_stored = 0
        self._total_filtered = 0
        
        logger.info(f"HierarchicalEventStorage initialized at {self.base_dir}")
    
    def set_context(self, workflow_id: str, phase_name: Optional[str] = None, 
                   container_id: Optional[str] = None) -> None:
        """Set current storage context for hierarchical organization."""
        self.workflow_id = workflow_id
        self.phase_name = phase_name
        self.container_id = container_id
        
        # Ensure directory exists
        path = self._get_current_path()
        path.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Storage context set: workflow={workflow_id}, phase={phase_name}, container={container_id}")
    
    def store(self, event: Event) -> None:
        """Store event with sparse filtering and batching."""
        # Apply sparse filtering
        if not self._should_store(event):
            self._total_filtered += 1
            return
        
        # Determine storage container
        container_id = event.container_id or self.container_id or 'root'
        
        # Buffer event
        self.event_buffers[container_id].append(event)
        self.buffer_sizes[container_id] += 1
        self._total_stored += 1
        
        # Update indices for fast lookup
        buffer_idx = len(self.event_buffers[container_id]) - 1
        event_id = event.metadata.get('event_id', event.event_id)
        
        if event.event_type == EventType.SIGNAL.value:
            self.signal_indices[container_id].append((buffer_idx, event_id))
        elif event.event_type in [EventType.POSITION_OPEN.value, EventType.POSITION_CLOSE.value]:
            self.trade_indices[container_id].append((buffer_idx, event_id))
        elif event.event_type == EventType.REGIME_CHANGE.value:
            self.regime_changes.append({
                'container_id': container_id,
                'timestamp': event.timestamp,
                'old_regime': event.payload.get('old_regime'),
                'new_regime': event.payload.get('new_regime'),
                'classifier': event.payload.get('classifier')
            })
        
        # Update statistics
        self.event_counts[container_id][event.event_type] += 1
        
        # Check if we should flush
        if self.buffer_sizes[container_id] >= self.config.batch_size:
            self.flush_container(container_id)
    
    def flush_container(self, container_id: str) -> None:
        """Flush events for a specific container to disk."""
        if not self.event_buffers[container_id]:
            return
        
        events = self.event_buffers[container_id]
        path = self._get_container_path(container_id)
        
        if self.config.format == 'parquet':
            self._write_parquet(events, path / 'events.parquet')
            
            # Write specialized indices
            self._write_signal_index(container_id, path)
            self._write_trade_index(container_id, path)
        else:
            self._write_jsonl(events, path / 'events.jsonl')
        
        # Write container metrics
        self._write_container_metrics(container_id, path)
        
        # Clear buffers
        self.event_buffers[container_id].clear()
        self.buffer_sizes[container_id] = 0
        
        logger.info(f"Flushed {len(events)} events for container {container_id}")
    
    def flush_all(self) -> None:
        """Flush all pending events."""
        for container_id in list(self.event_buffers.keys()):
            self.flush_container(container_id)
        
        # Write workflow summary
        if self.workflow_id:
            self._write_workflow_summary()
    
    def retrieve(self, event_id: str) -> Optional[Event]:
        """Retrieve event by ID using indices."""
        # Check in-memory buffers first
        for container_id, events in self.event_buffers.items():
            for event in events:
                if event.metadata.get('event_id') == event_id:
                    return event
        
        # TODO: Implement disk lookup using indices
        return None
    
    def query(self, criteria: Dict[str, Any]) -> List[Event]:
        """Query events with optimized lookup."""
        results = []
        
        # Optimize for common query patterns
        if 'container_id' in criteria:
            # Query specific container
            container_id = criteria['container_id']
            if container_id in self.event_buffers:
                results.extend(self._filter_events(self.event_buffers[container_id], criteria))
        
        elif 'event_type' in criteria and criteria['event_type'] == EventType.SIGNAL.value:
            # Use signal indices for fast lookup
            for container_id, indices in self.signal_indices.items():
                events = self.event_buffers.get(container_id, [])
                for idx, _ in indices:
                    if idx < len(events):
                        event = events[idx]
                        if self._matches_criteria(event, criteria):
                            results.append(event)
        
        else:
            # General query across all containers
            for events in self.event_buffers.values():
                results.extend(self._filter_events(events, criteria))
        
        return results
    
    def get_container_events(self, container_id: str) -> List[Event]:
        """Get all events for a specific container."""
        return self.event_buffers.get(container_id, []).copy()
    
    def get_signal_replay_data(self, container_id: Optional[str] = None) -> Dict[str, Any]:
        """Get signal data formatted for replay."""
        signals = []
        
        containers = [container_id] if container_id else self.signal_indices.keys()
        
        for cid in containers:
            events = self.event_buffers.get(cid, [])
            for idx, event_id in self.signal_indices.get(cid, []):
                if idx < len(events):
                    event = events[idx]
                    signals.append({
                        'container_id': cid,
                        'event_id': event_id,
                        'timestamp': event.timestamp.isoformat(),
                        'strategy_id': event.payload.get('strategy_id'),
                        'signal_value': event.payload.get('signal_value'),
                        'bar_data': event.payload.get('bars'),
                        'features': event.payload.get('features'),
                        'classifier_states': event.payload.get('classifier_states')
                    })
        
        return {
            'metadata': {
                'workflow_id': self.workflow_id,
                'phase_name': self.phase_name,
                'signal_count': len(signals),
                'containers': list(containers)
            },
            'signals': signals,
            'regime_changes': self.regime_changes.copy()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            'workflow_id': self.workflow_id,
            'phase_name': self.phase_name,
            'total_stored': self._total_stored,
            'total_filtered': self._total_filtered,
            'filter_ratio': self._total_filtered / max(1, self._total_stored + self._total_filtered),
            'containers': list(self.event_counts.keys()),
            'event_counts': dict(self.event_counts),
            'buffer_status': {
                cid: len(events) for cid, events in self.event_buffers.items()
            },
            'sparse_indices': {
                'signal_count': sum(len(idx) for idx in self.signal_indices.values()),
                'trade_count': sum(len(idx) for idx in self.trade_indices.values()),
                'regime_changes': len(self.regime_changes)
            }
        }
    
    # Private methods
    
    def _should_store(self, event: Event) -> bool:
        """Check if event should be stored (sparse filtering)."""
        # Always store configured sparse event types
        if event.event_type in self.config.sparse_event_types:
            return True
        
        # Store error events
        if 'error' in event.metadata:
            return True
        
        # Filter out high-frequency events unless explicitly configured
        high_freq_types = {EventType.BAR.value, EventType.TICK.value, EventType.FEATURES.value}
        if event.event_type in high_freq_types:
            return False
        
        return True
    
    def _get_current_path(self) -> Path:
        """Get current storage path based on context."""
        path = self.base_dir
        
        if self.workflow_id:
            path = path / self.workflow_id
        
        if self.phase_name and self.config.enable_phase_grouping:
            path = path / 'phases' / self.phase_name
        
        if self.container_id and self.config.enable_container_isolation:
            path = path / self.container_id
        
        return path
    
    def _get_container_path(self, container_id: str) -> Path:
        """Get path for specific container."""
        path = self.base_dir
        
        if self.workflow_id:
            path = path / self.workflow_id
        
        if self.phase_name:
            path = path / 'phases' / self.phase_name
        
        return path / container_id
    
    def _write_parquet(self, events: List[Event], filepath: Path) -> None:
        """Write events to Parquet format."""
        records = []
        for event in events:
            record = {
                'event_id': event.event_id,
                'event_type': event.event_type,
                'timestamp': event.timestamp,
                'container_id': event.container_id,
                'correlation_id': event.correlation_id,
                'causation_id': event.causation_id,
                'sequence_number': event.sequence_number,
                'payload': json.dumps(event.payload),
                'metadata': json.dumps(event.metadata)
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Write or append
        if filepath.exists():
            existing_df = pd.read_parquet(filepath)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_parquet(filepath, compression=self.config.compression)
    
    def _write_jsonl(self, events: List[Event], filepath: Path) -> None:
        """Write events to JSONL format."""
        with open(filepath, 'a') as f:
            for event in events:
                f.write(json.dumps(event.to_dict(), default=str) + '\n')
    
    def _write_signal_index(self, container_id: str, path: Path) -> None:
        """Write signal index for fast replay."""
        if container_id not in self.signal_indices:
            return
        
        signals = []
        events = self.event_buffers.get(container_id, [])
        
        for idx, event_id in self.signal_indices[container_id]:
            if idx < len(events):
                event = events[idx]
                signals.append({
                    'event_id': event_id,
                    'timestamp': event.timestamp,
                    'strategy_id': event.payload.get('strategy_id'),
                    'signal_value': event.payload.get('signal_value'),
                    'symbol': event.payload.get('symbol')
                })
        
        if signals:
            df = pd.DataFrame(signals)
            df.to_parquet(path / 'signals.parquet', compression=self.config.compression)
    
    def _write_trade_index(self, container_id: str, path: Path) -> None:
        """Write trade index for analysis."""
        if container_id not in self.trade_indices:
            return
        
        trades = []
        events = self.event_buffers.get(container_id, [])
        
        # Group by correlation_id to match trades
        trade_events = defaultdict(list)
        for idx, event_id in self.trade_indices[container_id]:
            if idx < len(events):
                event = events[idx]
                if event.correlation_id:
                    trade_events[event.correlation_id].append(event)
        
        # Create trade records
        for correlation_id, events_list in trade_events.items():
            open_event = next((e for e in events_list if e.event_type == EventType.POSITION_OPEN.value), None)
            close_event = next((e for e in events_list if e.event_type == EventType.POSITION_CLOSE.value), None)
            
            if open_event and close_event:
                trades.append({
                    'correlation_id': correlation_id,
                    'entry_time': open_event.timestamp,
                    'exit_time': close_event.timestamp,
                    'entry_price': open_event.payload.get('price'),
                    'exit_price': close_event.payload.get('price'),
                    'quantity': open_event.payload.get('quantity'),
                    'pnl': close_event.payload.get('pnl'),
                    'pnl_pct': close_event.payload.get('pnl_pct')
                })
        
        if trades:
            df = pd.DataFrame(trades)
            df.to_parquet(path / 'trades.parquet', compression=self.config.compression)
    
    def _write_container_metrics(self, container_id: str, path: Path) -> None:
        """Write container-specific metrics."""
        metrics = {
            'container_id': container_id,
            'event_counts': dict(self.event_counts.get(container_id, {})),
            'signal_count': len(self.signal_indices.get(container_id, [])),
            'trade_count': len(self.trade_indices.get(container_id, [])),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def _write_workflow_summary(self) -> None:
        """Write workflow-level summary."""
        summary = {
            'workflow_id': self.workflow_id,
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'regime_changes': self.regime_changes,
            'phase_summaries': {}
        }
        
        # Add phase summaries
        if self.phase_name:
            summary['phase_summaries'][self.phase_name] = {
                'containers': list(self.event_counts.keys()),
                'total_events': self._total_stored,
                'filtered_events': self._total_filtered
            }
        
        summary_path = self.base_dir / self.workflow_id / 'summary.json'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _filter_events(self, events: List[Event], criteria: Dict[str, Any]) -> List[Event]:
        """Filter events by criteria."""
        return [e for e in events if self._matches_criteria(e, criteria)]
    
    def _matches_criteria(self, event: Event, criteria: Dict[str, Any]) -> bool:
        """Check if event matches all criteria."""
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
    
    def prune(self, criteria: Dict[str, Any]) -> int:
        """Prune events matching criteria."""
        # Not implemented for hierarchical storage
        # Would require rewriting parquet files
        return 0
    
    def count(self) -> int:
        """Get total event count."""
        return self._total_stored
    
    def export_to_file(self, filepath: str) -> None:
        """Export all events to single file."""
        # Flush all buffers first
        self.flush_all()
        
        # TODO: Implement export across all parquet files
        pass
    
    def clear(self) -> None:
        """Clear all buffers."""
        self.event_buffers.clear()
        self.buffer_sizes.clear()
        self.signal_indices.clear()
        self.trade_indices.clear()
        self.regime_changes.clear()
        self.event_counts.clear()
        self._total_stored = 0
        self._total_filtered = 0