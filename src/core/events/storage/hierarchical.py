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

# Optional pyarrow dependency for parquet support
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    pa = None
    pq = None
    PARQUET_AVAILABLE = False
from dataclasses import dataclass, field

from ..protocols import EventStorageProtocol
from ..types import Event, EventType

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalStorageConfig:
    """Configuration for hierarchical event storage."""
    
    # Base directory for all workspaces
    base_dir: str = "./workspaces"
    
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
        EventType.RISK_BREACH.value,
        EventType.BAR.value  # Include BAR events for development/testing
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
    
    Directory structure (workspace-focused):
    workspaces/
    └── {workflow_id}/
        ├── metadata.json              # Workflow metadata
        ├── root_id/                   # Root/coordinator container
        │   ├── events.parquet
        │   └── metrics.json
        ├── portfolio1_id/             # Portfolio containers (primary parallelization)
        │   ├── events.parquet
        │   ├── signals.parquet
        │   ├── trades.parquet
        │   └── metrics.json
        ├── portfolio2_id/
        │   └── ...
        ├── execution_id/              # Execution container
        │   ├── events.parquet
        │   └── metrics.json
        └── analysis/
            ├── pattern_library.parquet
            └── optimization_results.json
    
    Note: Portfolio containers are the main focus for user analysis and parallelization.
    """
    
    def __init__(self, config: HierarchicalStorageConfig):
        """Initialize hierarchical storage."""
        self.config = config
        self.base_dir = Path(config.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if parquet is available and adjust format if needed
        if config.format == 'parquet' and not PARQUET_AVAILABLE:
            logger.warning("PyArrow not available, falling back to JSONL format")
            self.config.format = 'jsonl'
        
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
        logger.info("Using workspace structure: workspaces/{workflow_id}/{container_id}/")
    
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
        
        # Create workspace structure on first use
        if workflow_id:
            self._ensure_workspace_structure(workflow_id)
    
    def store(self, event: Event) -> None:
        """Store event with sparse filtering and batching."""
        # Apply sparse filtering
        if not self._should_store(event):
            self._total_filtered += 1
            logger.debug(f"Filtered out event {event.event_type} from container {event.container_id}")
            return
        
        # Determine storage container
        # Use context container_id if set (for incoming event tracing)
        # Otherwise use event's container_id (for published event tracing)
        container_id = self.container_id or event.container_id or 'root'
        
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
            logger.debug(f"No events to flush for container {container_id}")
            return
        
        events = self.event_buffers[container_id]
        path = self._get_container_path(container_id)
        
        # Ensure directory exists
        path.mkdir(parents=True, exist_ok=True)
        
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
        
        # Write workflow summary and portfolio summary
        if self.workflow_id:
            self._write_workflow_summary()
            self.save_portfolio_summary()
    
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
    
    def get_portfolio_containers(self) -> List[str]:
        """Get list of portfolio container IDs in this workspace."""
        portfolio_containers = []
        
        # Check metadata first
        if self.workflow_id:
            metadata_path = self.base_dir / self.workflow_id / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get('portfolio_containers', [])
        
        # Fall back to scanning container IDs
        for container_id in self.event_counts.keys():
            if self._identify_container_type(container_id) == 'portfolio':
                portfolio_containers.append(container_id)
        
        return portfolio_containers
    
    def get_workspace_info(self) -> Dict[str, Any]:
        """Get workspace summary information."""
        portfolio_containers = self.get_portfolio_containers()
        
        return {
            'workspace_id': self.workflow_id,
            'base_path': str(self.base_dir / self.workflow_id) if self.workflow_id else str(self.base_dir),
            'portfolio_containers': portfolio_containers,
            'total_containers': len(self.event_counts),
            'container_breakdown': {
                container_id: self._identify_container_type(container_id)
                for container_id in self.event_counts.keys()
            },
            'portfolio_focus': {
                'enabled': True,
                'count': len(portfolio_containers),
                'primary_access_pattern': 'parallelized_portfolio_analysis'
            }
        }
    
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
        """Get storage statistics with workspace focus."""
        portfolio_containers = self.get_portfolio_containers()
        
        return {
            'workspace_id': self.workflow_id,
            'phase_name': self.phase_name,
            'structure': 'workspaces/{workflow_id}/{container_id}/',
            'total_stored': self._total_stored,
            'total_filtered': self._total_filtered,
            'filter_ratio': self._total_filtered / max(1, self._total_stored + self._total_filtered),
            'containers': {
                'total': list(self.event_counts.keys()),
                'portfolio_containers': portfolio_containers,
                'portfolio_count': len(portfolio_containers)
            },
            'event_counts': dict(self.event_counts),
            'buffer_status': {
                cid: len(events) for cid, events in self.event_buffers.items()
            },
            'sparse_indices': {
                'signal_count': sum(len(idx) for idx in self.signal_indices.values()),
                'trade_count': sum(len(idx) for idx in self.trade_indices.values()),
                'regime_changes': len(self.regime_changes)
            },
            'portfolio_focus': {
                'primary_analysis_unit': 'portfolio_containers',
                'parallelization_ready': len(portfolio_containers) > 1,
                'analysis_path': f"workspaces/{self.workflow_id}/analysis/" if self.workflow_id else None
            }
        }
    
    def create_portfolio_summary(self) -> Dict[str, Any]:
        """Create accessible JSON summary of portfolio metrics for quick viewing."""
        portfolio_containers = self.get_portfolio_containers()
        
        summary = {
            'workspace_id': self.workflow_id,
            'created_at': datetime.now().isoformat(),
            'portfolio_count': len(portfolio_containers),
            'portfolios': {}
        }
        
        # Generate summary for each portfolio container
        for container_id in portfolio_containers:
            container_path = self._get_container_path(container_id)
            
            # Load container metrics if available
            metrics_path = container_path / 'metrics.json'
            container_metrics = {}
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    container_metrics = json.load(f)
            
            # Build portfolio summary
            portfolio_summary = {
                'container_id': container_id,
                'container_type': 'portfolio',
                'status': 'active' if container_id in self.event_buffers else 'completed',
                'event_summary': {
                    'total_events': sum(container_metrics.get('event_counts', {}).values()),
                    'signal_count': container_metrics.get('signal_count', 0),
                    'trade_count': container_metrics.get('trade_count', 0),
                    'event_breakdown': container_metrics.get('event_counts', {})
                },
                'files': {
                    'events': str(container_path / 'events.parquet') if (container_path / 'events.parquet').exists() else None,
                    'signals': str(container_path / 'signals.parquet') if (container_path / 'signals.parquet').exists() else None,
                    'trades': str(container_path / 'trades.parquet') if (container_path / 'trades.parquet').exists() else None,
                    'metrics': str(metrics_path) if metrics_path.exists() else None
                },
                'access_pattern': f"workspaces/{self.workflow_id}/{container_id}/",
                'last_updated': container_metrics.get('timestamp')
            }
            
            # Add performance metrics if trades exist
            trades_path = container_path / 'trades.parquet'
            if trades_path.exists():
                try:
                    trades_df = pd.read_parquet(trades_path)
                    if not trades_df.empty:
                        portfolio_summary['performance'] = {
                            'total_trades': len(trades_df),
                            'total_pnl': float(trades_df['pnl'].sum()) if 'pnl' in trades_df.columns else 0,
                            'win_rate': float((trades_df['pnl'] > 0).mean()) if 'pnl' in trades_df.columns else 0,
                            'avg_trade_pnl': float(trades_df['pnl'].mean()) if 'pnl' in trades_df.columns else 0,
                            'best_trade': float(trades_df['pnl'].max()) if 'pnl' in trades_df.columns else 0,
                            'worst_trade': float(trades_df['pnl'].min()) if 'pnl' in trades_df.columns else 0
                        }
                except Exception as e:
                    logger.warning(f"Could not load performance metrics for {container_id}: {e}")
            
            summary['portfolios'][container_id] = portfolio_summary
        
        return summary
    
    def save_portfolio_summary(self) -> Optional[str]:
        """Save portfolio summary to JSON file for easy access."""
        if not self.workflow_id:
            return None
        
        summary = self.create_portfolio_summary()
        summary_path = self.base_dir / self.workflow_id / 'portfolio_summary.json'
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Portfolio summary saved to {summary_path}")
        return str(summary_path)
    
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
        high_freq_types = {EventType.TICK.value, EventType.FEATURES.value}
        if event.event_type in high_freq_types:
            return False
        
        return True
    
    def _ensure_workspace_structure(self, workflow_id: str) -> None:
        """Ensure workspace directory structure exists."""
        workspace_path = self.base_dir / workflow_id
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Create analysis directory for cross-container results
        analysis_path = workspace_path / 'analysis'
        analysis_path.mkdir(exist_ok=True)
        
        # Create workspace metadata if it doesn't exist
        metadata_path = workspace_path / 'metadata.json'
        if not metadata_path.exists():
            metadata = {
                'workspace_id': workflow_id,
                'created_at': datetime.now().isoformat(),
                'description': 'ADMF-PC workspace for parallel portfolio analysis',
                'structure': 'workspaces/{workflow_id}/{container_id}/',
                'portfolio_containers': [],  # Will be populated as containers write
                'analysis_available': True
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def _identify_container_type(self, container_id: str) -> str:
        """Identify container type for workspace organization."""
        container_lower = container_id.lower()
        
        if 'portfolio' in container_lower or 'pf_' in container_lower:
            return 'portfolio'
        elif 'execution' in container_lower or 'exec_' in container_lower:
            return 'execution'
        elif 'root' in container_lower or 'coord' in container_lower:
            return 'root'
        elif 'data' in container_lower or 'feature' in container_lower:
            return 'data'
        elif 'strategy' in container_lower or 'signal' in container_lower:
            return 'strategy'
        elif 'risk' in container_lower:
            return 'risk'
        else:
            return 'other'
    
    def _get_current_path(self) -> Path:
        """Get current storage path based on context (workspace structure)."""
        path = self.base_dir
        
        if self.workflow_id:
            path = path / self.workflow_id
        
        # Skip phase grouping - use flat container structure for portfolio focus
        if self.container_id and self.config.enable_container_isolation:
            path = path / self.container_id
        
        return path
    
    def _get_container_path(self, container_id: str) -> Path:
        """Get path for specific container (workspace structure)."""
        path = self.base_dir
        
        if self.workflow_id:
            path = path / self.workflow_id
        
        # Flat structure: workspaces/{workflow_id}/{container_id}/
        return path / container_id
    
    def _write_parquet(self, events: List[Event], filepath: Path) -> None:
        """Write events to Parquet format."""
        if not PARQUET_AVAILABLE:
            logger.warning("PyArrow not available, falling back to JSONL")
            jsonl_path = filepath.with_suffix('.jsonl')
            self._write_jsonl(events, jsonl_path)
            return
            
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
            if PARQUET_AVAILABLE:
                df.to_parquet(path / 'signals.parquet', compression=self.config.compression)
            else:
                # Save as CSV instead
                df.to_csv(path / 'signals.csv', index=False)
    
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
            if PARQUET_AVAILABLE:
                df.to_parquet(path / 'trades.parquet', compression=self.config.compression)
            else:
                # Save as CSV instead
                df.to_csv(path / 'trades.csv', index=False)
    
    def _write_container_metrics(self, container_id: str, path: Path) -> None:
        """Write container-specific metrics and update workspace metadata."""
        container_type = self._identify_container_type(container_id)
        
        metrics = {
            'container_id': container_id,
            'container_type': container_type,
            'event_counts': dict(self.event_counts.get(container_id, {})),
            'signal_count': len(self.signal_indices.get(container_id, [])),
            'trade_count': len(self.trade_indices.get(container_id, [])),
            'timestamp': datetime.now().isoformat(),
            'is_portfolio_container': container_type == 'portfolio'
        }
        
        with open(path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Update workspace metadata with portfolio container info
        if self.workflow_id and container_type == 'portfolio':
            self._update_workspace_metadata(container_id)
    
    def _update_workspace_metadata(self, portfolio_container_id: str) -> None:
        """Update workspace metadata with portfolio container information."""
        if not self.workflow_id:
            return
            
        metadata_path = self.base_dir / self.workflow_id / 'metadata.json'
        
        # Load existing metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                'workspace_id': self.workflow_id,
                'created_at': datetime.now().isoformat(),
                'portfolio_containers': []
            }
        
        # Add portfolio container if not already present
        if portfolio_container_id not in metadata.get('portfolio_containers', []):
            metadata.setdefault('portfolio_containers', []).append(portfolio_container_id)
            metadata['last_updated'] = datetime.now().isoformat()
            metadata['total_portfolio_containers'] = len(metadata['portfolio_containers'])
            
            # Write updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
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