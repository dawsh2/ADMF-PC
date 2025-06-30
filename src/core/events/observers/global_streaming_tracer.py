"""
Global Streaming Multi-Strategy Tracer

Enhanced version of StreamingMultiStrategyTracer that implements global trace storage
with strategy hash-based file naming for cross-run deduplication.
"""

from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import logging
from datetime import datetime
import json
import hashlib
import os

from ..types import Event, EventType
from ..protocols import EventObserverProtocol
from ..storage.streaming_sparse_storage import StreamingSparseStorage
from ..storage.dense_event_storage import DenseEventStorage
from .strategy_metadata_extractor import update_metadata_with_recursive_strategies, compute_strategy_hash, extract_recursive_strategy_metadata

logger = logging.getLogger(__name__)


class GlobalStreamingTracer(EventObserverProtocol):
    """
    Global multi-strategy tracer with hash-based trace storage.
    
    Key features:
    1. Writes traces to global /traces directory at project root
    2. Uses strategy hashes for file naming to enable deduplication
    3. Creates queryable strategy_index.parquet
    4. Maintains backward compatibility with workspace metadata
    """
    
    def __init__(self, 
                 workspace_path: str,
                 workflow_id: str,
                 managed_strategies: Optional[List[str]] = None,
                 managed_classifiers: Optional[List[str]] = None,
                 data_source_config: Optional[Dict[str, Any]] = None,
                 write_interval: int = 0,
                 write_on_changes: int = 0,
                 full_config: Optional[Dict[str, Any]] = None,
                 global_traces_root: Optional[str] = None):
        """
        Initialize global streaming tracer.
        
        Args:
            workspace_path: Run-specific directory for metadata
            workflow_id: Workflow identifier
            managed_strategies: Strategies to trace
            managed_classifiers: Classifiers to trace
            data_source_config: Data source configuration
            write_interval: Write to disk every N bars (0 = only at end)
            write_on_changes: Write to disk every M changes (0 = only at end)
            full_config: Full configuration dict containing strategy definitions
            global_traces_root: Override global traces directory (defaults to project root/traces)
        """
        self._workspace_path = Path(workspace_path)
        self._workflow_id = workflow_id
        self._data_source_config = data_source_config or {}
        self._write_interval = write_interval
        self._write_on_changes = write_on_changes
        self._full_config = full_config or {}
        
        # If not specified, trace all
        self._managed_strategies = set(managed_strategies) if managed_strategies else None
        self._managed_classifiers = set(managed_classifiers) if managed_classifiers else None
        
        # Storage instances per component (sparse for signals)
        self._storages: Dict[str, StreamingSparseStorage] = {}
        
        # Dense storage for orders, fills, and positions
        self._dense_storages: Dict[str, DenseEventStorage] = {}
        
        # Current bar count - 0-based to match source files
        self._current_bar_count = 0
        self._initialized_bar_count = False
        
        # Metrics
        self._total_signals = 0
        self._total_classifications = 0
        self._total_orders = 0
        self._total_fills = 0
        self._total_positions = 0
        self._stored_changes = 0
        
        # Component metadata and hash mapping
        self._component_metadata: Dict[str, Dict[str, Any]] = {}
        self._strategy_hash_to_path: Dict[str, Path] = {}
        self._existing_hashes: Set[str] = set()
        
        # Determine global traces directory
        if global_traces_root:
            self._global_traces_dir = Path(global_traces_root)
        else:
            # Find project root by looking for a marker file (e.g., .git, pyproject.toml)
            current = Path.cwd()
            while current != current.parent:
                if (current / '.git').exists() or (current / 'pyproject.toml').exists():
                    self._global_traces_dir = current / 'traces'
                    break
                current = current.parent
            else:
                # Fallback to cwd/traces if project root not found
                self._global_traces_dir = Path.cwd() / 'traces'
        
        # Create global traces directory
        self._global_traces_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing strategy hashes to prevent redundant computation
        self._load_existing_hashes()
        
        logger.info(f"GlobalStreamingTracer initialized")
        logger.info(f"  Workspace (metadata): {workspace_path}")
        logger.info(f"  Global traces: {self._global_traces_dir}")
        if write_interval > 0 or write_on_changes > 0:
            logger.info(f"  Write settings: every {write_interval} bars or {write_on_changes} changes")
        else:
            logger.info("  Write settings: only at end (no periodic writes)")
    
    def _load_existing_hashes(self):
        """Load existing strategy hashes from global traces directory."""
        strategy_index_path = self._global_traces_dir / 'strategy_index.parquet'
        if strategy_index_path.exists():
            try:
                import pandas as pd
                index_df = pd.read_parquet(strategy_index_path)
                self._existing_hashes = set(index_df['strategy_hash'].unique())
                logger.info(f"Loaded {len(self._existing_hashes)} existing strategy hashes")
            except Exception as e:
                logger.warning(f"Could not load existing strategy index: {e}")
    
    def has_strategy_hash(self, strategy_hash: str) -> bool:
        """Check if a strategy hash already exists in global traces."""
        return strategy_hash in self._existing_hashes
    
    def on_event(self, event: Event) -> None:
        """Process events from the root event bus."""
        # Debug position events
        if "POSITION" in event.event_type:
            logger.info(f"GlobalStreamingTracer.on_event: Got {event.event_type} event")
        
        if event.event_type == EventType.BAR.value:
            # Use original bar index from event payload for consistent sparse storage
            has_payload = hasattr(event, 'payload')
            has_original_index = has_payload and event.payload and 'original_bar_index' in event.payload
            
            if has_original_index:
                self._current_bar_count = event.payload['original_bar_index']
            else:
                # Fallback to incrementing for backward compatibility
                if not self._initialized_bar_count:
                    self._current_bar_count = -1
                    self._initialized_bar_count = True
                self._current_bar_count += 1
            
            # Log progress periodically
            if self._current_bar_count % 1000 == 0:
                logger.debug(f"Processed {self._current_bar_count} bars, "
                            f"{self._stored_changes} signal changes stored")
            
        elif event.event_type == EventType.SIGNAL.value:
            self._process_signal_event(event)
            
        elif event.event_type == EventType.CLASSIFICATION.value:
            self._process_classification_event(event)
            
        elif event.event_type == EventType.ORDER.value:
            self._process_order_event(event)
            
        elif event.event_type == EventType.FILL.value:
            self._process_fill_event(event)
            
        elif event.event_type in [EventType.POSITION_OPEN.value, EventType.POSITION_CLOSE.value]:
            logger.debug(f"GlobalStreamingTracer received {event.event_type} event")
            self._process_position_event(event)
    
    def _process_signal_event(self, event: Event) -> None:
        """Process SIGNAL events from strategies."""
        payload = event.payload
        strategy_id = payload.get('strategy_id', '')
        
        # Check if we should trace this strategy
        if self._managed_strategies and strategy_id not in self._managed_strategies:
            is_managed = any(ms in strategy_id for ms in self._managed_strategies)
            if not is_managed:
                return
        
        self._total_signals += 1
        
        # Get or create storage for this strategy
        storage = self._get_or_create_storage('strategy', strategy_id, payload)
        
        # Extract full metadata and compute hash (only done once per strategy)
        full_metadata = None
        strategy_hash = None
        
        if strategy_id not in self._component_metadata or 'strategy_hash' not in self._component_metadata.get(strategy_id, {}):
            # Build complete metadata including parameters and configuration
            strategy_type = payload.get('strategy_type', 'unknown')
            parameters = payload.get('parameters', payload.get('metadata', {}).get('parameters', {}))
            
            # For full config, check if we have it in payload
            strategy_config = payload.get('strategy_config')
            
            # If not in payload, build it from available data
            if not strategy_config:
                strategy_config = {
                    'type': strategy_type,
                    'parameters': parameters
                }
                
                # Add constraints if present in payload
                if 'constraints' in payload:
                    strategy_config['constraints'] = payload['constraints']
                elif 'threshold' in payload:
                    strategy_config['constraints'] = payload['threshold']
            
            full_metadata = strategy_config
            strategy_hash = compute_strategy_hash(strategy_config)
            
            # Update component metadata with hash
            if strategy_id in self._component_metadata:
                self._component_metadata[strategy_id]['strategy_hash'] = strategy_hash
                self._component_metadata[strategy_id]['full_config'] = strategy_config
        
        # Process the signal
        direction = payload.get('direction', 'flat')
        symbol = payload.get('symbol', 'UNKNOWN')
        
        # Use bar timestamp from payload instead of execution timestamp
        bar_timestamp = payload.get('timestamp')
        if bar_timestamp:
            timestamp = bar_timestamp.isoformat() if hasattr(bar_timestamp, 'isoformat') else str(bar_timestamp)
        else:
            timestamp = event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp)
        
        price = payload.get('price', payload.get('metadata', {}).get('price', 0.0))
        signal_metadata = payload.get('metadata', {})
        
        was_change = storage.process_signal(
            symbol=symbol,
            direction=str(direction),
            strategy_id=strategy_id,
            timestamp=timestamp,
            price=price,
            bar_index=self._current_bar_count,
            metadata=signal_metadata,
            strategy_hash=strategy_hash
        )
        
        if was_change:
            self._stored_changes += 1
    
    def _process_classification_event(self, event: Event) -> None:
        """Process CLASSIFICATION events from classifiers."""
        payload = event.payload
        classifier_id = payload.get('classifier_id', '')
        
        # Check if we should trace this classifier
        if self._managed_classifiers and classifier_id not in self._managed_classifiers:
            is_managed = any(mc in classifier_id for mc in self._managed_classifiers)
            if not is_managed:
                return
        
        self._total_classifications += 1
        
        # Get or create storage for this classifier
        storage = self._get_or_create_storage('classifier', classifier_id, payload)
        
        # Process the classification
        regime = payload.get('regime', 'unknown')
        symbol = payload.get('symbol', 'UNKNOWN')
        
        # Use bar timestamp from payload
        bar_timestamp = payload.get('timestamp')
        if bar_timestamp:
            timestamp = bar_timestamp.isoformat() if hasattr(bar_timestamp, 'isoformat') else str(bar_timestamp)
        else:
            timestamp = event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp)
        
        was_change = storage.process_signal(
            symbol=symbol,
            direction=str(regime),
            strategy_id=classifier_id,
            timestamp=timestamp,
            price=0.0,
            bar_index=self._current_bar_count
        )
        
        if was_change:
            self._stored_changes += 1
    
    def _process_order_event(self, event: Event) -> None:
        """Process ORDER events from portfolio."""
        payload = event.payload
        symbol = payload.get('symbol', 'UNKNOWN')
        timestamp = event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp)
        
        self._total_orders += 1
        
        # Store orders in workspace directory for run-specific data
        storage_key = 'portfolio_orders'
        if storage_key not in self._dense_storages:
            orders_dir = self._workspace_path / 'traces' / 'portfolio' / 'orders'
            orders_dir.mkdir(parents=True, exist_ok=True)
            self._dense_storages[storage_key] = DenseEventStorage(str(orders_dir), 'orders')
        
        self._dense_storages[storage_key].add_event(
            symbol=symbol,
            timestamp=timestamp,
            bar_index=self._current_bar_count,
            metadata=payload
        )
        
        order_id = payload.get('order_id', 'unknown')
        logger.debug(f"Stored ORDER event: {order_id} for {symbol}")
    
    def _process_fill_event(self, event: Event) -> None:
        """Process FILL events from execution engine."""
        payload = event.payload
        symbol = payload.get('symbol', 'UNKNOWN')
        timestamp = event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp)
        
        self._total_fills += 1
        
        # Store fills in workspace directory for run-specific data
        storage_key = 'execution_fills'
        if storage_key not in self._dense_storages:
            fills_dir = self._workspace_path / 'traces' / 'execution' / 'fills'
            fills_dir.mkdir(parents=True, exist_ok=True)
            self._dense_storages[storage_key] = DenseEventStorage(str(fills_dir), 'fills')
        
        self._dense_storages[storage_key].add_event(
            symbol=symbol,
            timestamp=timestamp,
            bar_index=self._current_bar_count,
            metadata=payload
        )
        
        fill_id = payload.get('fill_id', 'unknown')
        logger.debug(f"Stored FILL event: {fill_id} for {symbol}")
    
    def _process_position_event(self, event: Event) -> None:
        """Process POSITION_OPEN/CLOSE events."""
        payload = event.payload
        symbol = payload.get('symbol', 'UNKNOWN')
        timestamp = event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp)
        
        self._total_positions += 1
        
        # Store positions in workspace directory for run-specific data
        if event.event_type == EventType.POSITION_OPEN.value:
            storage_key = 'positions_open'
            dir_name = 'positions_open'
        else:
            storage_key = 'positions_close'
            dir_name = 'positions_close'
        
        if storage_key not in self._dense_storages:
            positions_dir = self._workspace_path / 'traces' / 'portfolio' / dir_name
            positions_dir.mkdir(parents=True, exist_ok=True)
            self._dense_storages[storage_key] = DenseEventStorage(str(positions_dir), dir_name)
        
        self._dense_storages[storage_key].add_event(
            symbol=symbol,
            timestamp=timestamp,
            bar_index=self._current_bar_count,
            metadata=payload
        )
        
        logger.debug(f"Stored {event.event_type} event for {symbol} in {storage_key} storage")
    
    def _get_or_create_storage(self, component_type: str, component_id: str, 
                              payload: Dict[str, Any]) -> StreamingSparseStorage:
        """Get or create a storage instance for a component."""
        if component_id in self._storages:
            return self._storages[component_id]
        
        # Extract symbol and timeframe
        symbol = payload.get('symbol', 'UNKNOWN')
        timeframe = payload.get('timeframe', '1m')
        
        # Extract strategy/classifier type
        if component_type == 'strategy':
            strategy_type = payload.get('strategy_type')
        else:
            strategy_type = payload.get('classifier_type')
        
        # Fallback to legacy extraction from component_id if not in payload
        if not strategy_type:
            component_parts = component_id.split('_')
            if 'grid' in component_id:
                grid_idx = component_parts.index('grid')
                start_idx = 1 if component_parts[0] == symbol else 0
                strategy_type = '_'.join(component_parts[start_idx:grid_idx + 1])
            else:
                start_idx = 1 if component_parts[0] == symbol else 0
                strategy_type = component_parts[start_idx] if start_idx < len(component_parts) else 'unknown'
        
        # Extract parameters and compute hash
        parameters = payload.get('parameters', {})
        if not parameters and 'metadata' in payload:
            parameters = payload.get('metadata', {}).get('parameters', {})
        
        # Build strategy config for hashing
        strategy_config = {
            'type': strategy_type,
            'parameters': parameters
        }
        
        # Add constraints if present
        if 'constraints' in payload:
            strategy_config['constraints'] = payload['constraints']
        elif 'threshold' in payload:
            strategy_config['constraints'] = payload['threshold']
        
        # Compute strategy hash
        strategy_hash = compute_strategy_hash(strategy_config)
        
        # Create global directory structure: traces/SYMBOL/TIMEFRAME/COMPONENT_TYPE/STRATEGY_TYPE/
        if component_type == 'strategy':
            component_dir = self._global_traces_dir / symbol / timeframe / 'signals' / strategy_type
        else:
            component_dir = self._global_traces_dir / symbol / timeframe / 'classifiers' / strategy_type
        
        # Use hash-based filename
        filename = f"{strategy_hash}.parquet"
        
        # Check if this hash already exists
        if strategy_hash in self._existing_hashes:
            logger.debug(f"Strategy {strategy_type} ({strategy_hash}) already exists in global traces, skipping computation")
            # Still create storage to track the reference, but it won't actually write
            # TODO: In future, we could return a read-only reference instead
        
        # Create streaming storage with hash-based filename
        storage = StreamingSparseStorage(
            base_dir=str(component_dir),
            write_interval=self._write_interval,
            write_on_changes=self._write_on_changes,
            component_id=strategy_hash  # Use hash as filename
        )
        
        self._storages[component_id] = storage
        self._strategy_hash_to_path[strategy_hash] = component_dir / filename
        
        # Store metadata
        if component_id not in self._component_metadata:
            self._component_metadata[component_id] = {
                'type': component_type,
                'strategy_type': strategy_type,
                'strategy_hash': strategy_hash,
                'symbol': symbol,
                'timeframe': timeframe,
                'parameters': parameters,
                'trace_path': str(component_dir / filename)  # Global path
            }
        
        logger.debug(f"Created global storage for {component_type} {component_id} at {component_dir} (hash: {strategy_hash})")
        
        return storage
    
    def finalize(self) -> Dict[str, Any]:
        """Finalize all storages and save metadata."""
        logger.info(f"Finalizing GlobalStreamingTracer...")
        
        # Save dense event storages (orders, fills, positions) in workspace
        for storage_key, dense_storage in self._dense_storages.items():
            filepath = dense_storage.save(f"{storage_key}.parquet")
            if filepath:
                logger.info(f"Saved dense storage {storage_key} to {filepath}")
        
        # Finalize all sparse storages (signals) in global traces
        for component_id, storage in self._storages.items():
            try:
                storage.finalize()
            except Exception as e:
                logger.error(f"Error finalizing storage for {component_id}: {e}")
        
        # Save workspace metadata (backward compatibility)
        metadata = {
            'workflow_id': self._workflow_id,
            'workspace_path': str(self._workspace_path),
            'global_traces_path': str(self._global_traces_dir),
            'data_source': self._data_source_config,
            'total_bars': self._current_bar_count,
            'total_signals': self._total_signals,
            'total_classifications': self._total_classifications,
            'total_orders': self._total_orders,
            'total_fills': self._total_fills,
            'total_positions': self._total_positions,
            'stored_changes': self._stored_changes,
            'compression_ratio': (self._stored_changes / self._total_signals * 100) if self._total_signals > 0 else 0,
            'components': self._component_metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add mode if available in full config
        if self._full_config and 'mode' in self._full_config:
            metadata['mode'] = self._full_config['mode']
        
        # Add recursive strategy metadata if full config was provided
        if self._full_config:
            try:
                metadata = update_metadata_with_recursive_strategies(metadata, self._full_config)
                logger.debug("Added recursive strategy metadata to results")
            except Exception as e:
                logger.error(f"Failed to extract recursive strategy metadata: {e}")
        
        # Save metadata in workspace (create directory if needed)
        self._workspace_path.mkdir(parents=True, exist_ok=True)
        metadata_path = self._workspace_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update global strategy index
        self._update_global_strategy_index()
        
        # Create run-specific strategy index in workspace
        self._create_workspace_strategy_index()
        
        logger.info(f"Workspace metadata: {self._workspace_path}")
        logger.info(f"Global traces: {self._global_traces_dir}")
        
        # Calculate compression ratio safely
        if self._total_signals > 0:
            compression_pct = self._stored_changes / self._total_signals * 100
            logger.info(f"Total signals: {self._total_signals}, "
                       f"Changes stored: {self._stored_changes} "
                       f"({compression_pct:.1f}% compression)")
        else:
            logger.info(f"No signals generated (0 signals)")
        
        return metadata
    
    def _update_global_strategy_index(self) -> None:
        """Update the global strategy index with new strategies."""
        import pandas as pd
        
        # Load existing index if it exists
        index_path = self._global_traces_dir / 'strategy_index.parquet'
        if index_path.exists():
            existing_df = pd.read_parquet(index_path)
            existing_hashes = set(existing_df['strategy_hash'].unique())
        else:
            existing_df = None
            existing_hashes = set()
        
        # Build new entries
        new_entries = []
        for strategy_id, metadata in self._component_metadata.items():
            if metadata.get('type') != 'strategy':
                continue
            
            strategy_hash = metadata.get('strategy_hash', '')
            if strategy_hash and strategy_hash not in existing_hashes:
                # Extract parameters for easy querying
                params = metadata.get('parameters', {})
                
                entry = {
                    'strategy_hash': strategy_hash,
                    'strategy_type': metadata.get('strategy_type', 'unknown'),
                    'symbol': metadata.get('symbol', ''),
                    'timeframe': metadata.get('timeframe', ''),
                    'constraints': metadata.get('full_config', {}).get('constraints'),
                    'trace_path': metadata.get('trace_path', ''),
                    'first_seen': datetime.now().isoformat(),
                }
                
                # Add parameters as direct columns
                for param_name, param_value in params.items():
                    if not param_name.startswith('_'):
                        entry[param_name] = param_value
                
                new_entries.append(entry)
        
        if new_entries:
            new_df = pd.DataFrame(new_entries)
            
            # Combine with existing if present
            if existing_df is not None:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            
            # Save updated index
            combined_df.to_parquet(index_path, engine='pyarrow', index=False)
            logger.info(f"Updated global strategy index with {len(new_entries)} new strategies")
            logger.info(f"Total unique strategies in global index: {len(combined_df)}")
    
    def _create_workspace_strategy_index(self) -> None:
        """Create a strategy index for this specific run."""
        import pandas as pd
        
        index_data = []
        
        for strategy_id, metadata in self._component_metadata.items():
            if metadata.get('type') != 'strategy':
                continue
            
            # Extract parameters for easy querying
            params = metadata.get('parameters', {})
            strategy_hash = metadata.get('strategy_hash', '')
            
            entry = {
                'strategy_id': strategy_id,
                'strategy_hash': strategy_hash,
                'strategy_type': metadata.get('strategy_type', 'unknown'),
                'symbol': metadata.get('symbol', ''),
                'timeframe': metadata.get('timeframe', ''),
                'constraints': metadata.get('full_config', {}).get('constraints'),
                'global_trace_path': metadata.get('trace_path', ''),
            }
            
            # Add parameters as direct columns
            for param_name, param_value in params.items():
                if not param_name.startswith('_'):
                    entry[param_name] = param_value
            
            index_data.append(entry)
        
        if index_data:
            # Create DataFrame and save as parquet
            index_df = pd.DataFrame(index_data)
            
            # Log what we're creating
            logger.info(f"Creating workspace strategy index with {len(index_data)} strategies")
            logger.info(f"Unique hashes in this run: {index_df['strategy_hash'].nunique()}")
            
            index_path = self._workspace_path / 'strategy_index.parquet'
            index_df.to_parquet(index_path, engine='pyarrow', index=False)
            logger.info(f"Saved workspace strategy index to {index_path}")
        else:
            logger.warning("No strategies found for index creation")
    
    # Required protocol methods
    def on_publish(self, event: Event) -> None:
        """Called when event is published."""
        self.on_event(event)
    
    def on_delivered(self, event: Event, handler: Any) -> None:
        """Called when event is delivered."""
        pass
    
    def on_error(self, event: Event, handler: Any, error: Exception) -> None:
        """Called when handler raises error."""
        logger.error(f"Handler error for {event.event_type}: {error}")