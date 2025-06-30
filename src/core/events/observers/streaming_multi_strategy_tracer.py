"""
Streaming Multi-Strategy Tracer

Enhanced version of MultiStrategyTracer that uses streaming sparse storage
to prevent memory accumulation during long runs.
"""

from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import logging
from datetime import datetime
import json
import hashlib

from ..types import Event, EventType
from ..protocols import EventObserverProtocol
from ..storage.streaming_sparse_storage import StreamingSparseStorage
from ..storage.dense_event_storage import DenseEventStorage
from .strategy_metadata_extractor import update_metadata_with_recursive_strategies, compute_strategy_hash, extract_recursive_strategy_metadata

logger = logging.getLogger(__name__)


class StreamingMultiStrategyTracer(EventObserverProtocol):
    """
    Multi-strategy tracer with streaming writes to prevent memory issues.
    
    Key improvements:
    1. Uses StreamingSparseStorage that writes periodically
    2. Configurable write intervals
    3. Memory-efficient for long runs
    """
    
    def __init__(self, 
                 workspace_path: str,
                 workflow_id: str,
                 managed_strategies: Optional[List[str]] = None,
                 managed_classifiers: Optional[List[str]] = None,
                 data_source_config: Optional[Dict[str, Any]] = None,
                 write_interval: int = 0,
                 write_on_changes: int = 0,
                 full_config: Optional[Dict[str, Any]] = None):
        """
        Initialize streaming multi-strategy tracer.
        
        Args:
            workspace_path: Base directory for signal storage
            workflow_id: Workflow identifier
            managed_strategies: Strategies to trace
            managed_classifiers: Classifiers to trace
            data_source_config: Data source configuration
            write_interval: Write to disk every N bars (0 = only at end)
            write_on_changes: Write to disk every M changes (0 = only at end)
            full_config: Full configuration dict containing strategy definitions for metadata extraction
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
        
        # Component metadata
        self._component_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Create base traces directory
        self._traces_dir = self._workspace_path / "traces"
        self._traces_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"StreamingMultiStrategyTracer initialized for workspace: {workspace_path}")
        if write_interval > 0 or write_on_changes > 0:
            logger.info(f"Write settings: every {write_interval} bars or {write_on_changes} changes")
        else:
            logger.info("Write settings: only at end (no periodic writes)")
    
    def on_event(self, event: Event) -> None:
        """Process events from the root event bus."""
        # Debug position events
        if "POSITION" in event.event_type:
            logger.info(f"StreamingMultiStrategyTracer.on_event: Got {event.event_type} event")
        
        if event.event_type == EventType.BAR.value:
            # Use original bar index from event payload for consistent sparse storage
            has_payload = hasattr(event, 'payload')
            has_original_index = has_payload and event.payload and 'original_bar_index' in event.payload
            
            if has_original_index:
                self._current_bar_count = event.payload['original_bar_index']  # Keep 0-based indexing to match source
            else:
                # Debug why we're falling back
                if not has_payload:
                    logger.warning("⚠️  BAR event missing payload - using incremental count")
                elif not event.payload:
                    logger.warning("⚠️  BAR event payload is None - using incremental count")
                else:
                    logger.warning(f"⚠️  BAR event missing original_bar_index - payload keys: {list(event.payload.keys())} - using incremental count")
                
                # Fallback to incrementing for backward compatibility
                # Start from -1 so first increment gives 0
                if not hasattr(self, '_initialized_bar_count'):
                    self._current_bar_count = -1
                    self._initialized_bar_count = True
                self._current_bar_count += 1
            
            # Log progress periodically (reduced frequency for performance)
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
            logger.debug(f"StreamingMultiStrategyTracer received {event.event_type} event")
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
            # Log what we're hashing for debugging
            logger.debug(f"Computing hash for strategy {strategy_id} with config: {strategy_config}")
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
            # Fallback to event timestamp for backward compatibility
            timestamp = event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp)
        # Look for price at top level first, then in metadata
        price = payload.get('price', payload.get('metadata', {}).get('price', 0.0))
        
        # Get the signal's metadata (contains OHLC data), not the strategy config
        signal_metadata = payload.get('metadata', {})
        
        was_change = storage.process_signal(
            symbol=symbol,
            direction=str(direction),
            strategy_id=strategy_id,
            timestamp=timestamp,
            price=price,
            bar_index=self._current_bar_count,
            metadata=signal_metadata,  # Pass signal metadata with OHLC
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
        
        # Use bar timestamp from payload instead of execution timestamp
        bar_timestamp = payload.get('timestamp')
        if bar_timestamp:
            timestamp = bar_timestamp.isoformat() if hasattr(bar_timestamp, 'isoformat') else str(bar_timestamp)
        else:
            # Fallback to event timestamp for backward compatibility
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
        
        # Get or create dense storage for orders
        storage_key = 'portfolio_orders'
        if storage_key not in self._dense_storages:
            orders_dir = self._traces_dir / 'portfolio' / 'orders'
            self._dense_storages[storage_key] = DenseEventStorage(str(orders_dir), 'orders')
        
        # Store every order event
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
        
        # Get or create dense storage for fills
        storage_key = 'execution_fills'
        if storage_key not in self._dense_storages:
            fills_dir = self._traces_dir / 'execution' / 'fills'
            self._dense_storages[storage_key] = DenseEventStorage(str(fills_dir), 'fills')
        
        # Store every fill event
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
        logger.debug(f"Processing {event.event_type} for {symbol}, total positions: {self._total_positions}")
        
        # Store position opens and closes in separate dense storages
        if event.event_type == EventType.POSITION_OPEN.value:
            storage_key = 'positions_open'
            dir_name = 'positions_open'
        else:
            storage_key = 'positions_close'
            dir_name = 'positions_close'
        
        # Get or create dense storage for this position event type
        if storage_key not in self._dense_storages:
            positions_dir = self._traces_dir / 'portfolio' / dir_name
            self._dense_storages[storage_key] = DenseEventStorage(str(positions_dir), dir_name)
        
        # Store every position event
        self._dense_storages[storage_key].add_event(
            symbol=symbol,
            timestamp=timestamp,
            bar_index=self._current_bar_count,
            metadata=payload
        )
        
        logger.debug(f"Stored {event.event_type} event for {symbol} in {storage_key} storage")
        logger.debug(f"Dense storage {storage_key} now has {len(self._dense_storages[storage_key]._events)} events")
    
    def _get_or_create_storage(self, component_type: str, component_id: str, 
                              payload: Dict[str, Any]) -> StreamingSparseStorage:
        """Get or create a storage instance for a component."""
        if component_id in self._storages:
            return self._storages[component_id]
        
        # Extract symbol and timeframe
        symbol = payload.get('symbol', 'UNKNOWN')
        timeframe = payload.get('timeframe', '1m')
        
        # Extract strategy/classifier type from structured payload
        # First try to get from structured event payload (v2.0)
        if component_type == 'strategy':
            strategy_type = payload.get('strategy_type')
        else:  # classifier
            strategy_type = payload.get('classifier_type')
        
        # Fallback to legacy extraction from component_id if not in payload
        if not strategy_type:
            component_parts = component_id.split('_')
            if 'grid' in component_id:
                # Find the grid pattern (e.g., ma_crossover_grid, rsi_grid)
                grid_idx = component_parts.index('grid')
                start_idx = 1 if component_parts[0] == symbol else 0
                strategy_type = '_'.join(component_parts[start_idx:grid_idx + 1])  # Include 'grid'
            else:
                start_idx = 1 if component_parts[0] == symbol else 0
                strategy_type = component_parts[start_idx] if start_idx < len(component_parts) else 'unknown'
        
        # Create subdirectory structure: traces/signals|classifiers/STRATEGY_TYPE/
        # (symbol and timeframe are already in the filename and metadata)
        if component_type == 'strategy':
            component_dir = self._traces_dir / 'signals' / strategy_type
        else:
            component_dir = self._traces_dir / 'classifiers' / strategy_type
        
        # Create streaming storage with component_id for filename
        storage = StreamingSparseStorage(
            base_dir=str(component_dir),
            write_interval=self._write_interval,
            write_on_changes=self._write_on_changes,
            component_id=component_id
        )
        
        self._storages[component_id] = storage
        
        # Store metadata
        if component_id not in self._component_metadata:
            # Extract parameters from payload or metadata
            parameters = payload.get('parameters', {})
            if not parameters and 'metadata' in payload:
                parameters = payload.get('metadata', {}).get('parameters', {})
            
            # Debug logging
            if not parameters:
                logger.warning(f"[STREAMING_TRACER] No parameters found for {component_id}")
                logger.debug(f"[STREAMING_TRACER] Payload keys: {list(payload.keys())}")
                if 'metadata' in payload:
                    logger.debug(f"[STREAMING_TRACER] Metadata keys: {list(payload['metadata'].keys())}")
            else:
                logger.info(f"[STREAMING_TRACER] Found parameters for {component_id}: {list(parameters.keys())}")
            
            self._component_metadata[component_id] = {
                'type': component_type,
                'strategy_type': strategy_type,
                'symbol': symbol,
                'timeframe': timeframe,
                'parameters': parameters
            }
        
        logger.debug(f"Created streaming storage for {component_type} {component_id} at {component_dir}")
        
        return storage
    
    def finalize(self) -> Dict[str, Any]:
        """Finalize all storages and save metadata."""
        logger.info(f"Finalizing StreamingMultiStrategyTracer...")
        
        # Save dense event storages first (orders, fills, positions)
        for storage_key, dense_storage in self._dense_storages.items():
            filepath = dense_storage.save(f"{storage_key}.parquet")
            if filepath:
                logger.info(f"Saved dense storage {storage_key} to {filepath}")
        
        # Finalize all sparse storages (signals)
        for component_id, storage in self._storages.items():
            try:
                storage.finalize()
            except Exception as e:
                logger.error(f"Error finalizing storage for {component_id}: {e}")
        
        # Save workspace metadata
        metadata = {
            'workflow_id': self._workflow_id,
            'workspace_path': str(self._workspace_path),  # Include workspace path for analytics
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
        
        # Add recursive strategy metadata if full config was provided
        if self._full_config:
            try:
                metadata = update_metadata_with_recursive_strategies(metadata, self._full_config)
                logger.debug("Added recursive strategy metadata to results")
            except Exception as e:
                logger.error(f"Failed to extract recursive strategy metadata: {e}")
                # Continue without recursive metadata rather than failing
        
        metadata_path = self._workspace_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create strategy index
        self._create_strategy_index()
        
        logger.info(f"Workspace finalized: {self._workspace_path}")
        
        # Calculate compression ratio safely
        if self._total_signals > 0:
            compression_pct = self._stored_changes / self._total_signals * 100
            logger.info(f"Total signals: {self._total_signals}, "
                       f"Changes stored: {self._stored_changes} "
                       f"({compression_pct:.1f}% compression)")
        else:
            logger.info(f"No signals generated (0 signals)")
        
        return metadata
    
    def _create_strategy_index(self) -> None:
        """Create a queryable index of all strategies in this run.
        
        Following trace-updates.md specification:
        - Parameters stored as direct columns (period, std_dev, etc)
        - Unique strategy hash per configuration
        - Self-documenting with embedded metadata
        """
        import pandas as pd
        
        index_data = []
        
        for strategy_id, metadata in self._component_metadata.items():
            if metadata.get('type') != 'strategy':
                continue
                
            # Extract parameters for easy querying
            params = metadata.get('parameters', {})
            
            # Ensure unique hash based on actual parameters
            strategy_hash = metadata.get('strategy_hash', '')
            
            # Always compute hash from actual parameters to ensure uniqueness
            hash_config = {
                'type': metadata.get('strategy_type', 'unknown'),
                'parameters': {k: v for k, v in params.items() if not k.startswith('_')}
            }
            config_str = json.dumps(hash_config, sort_keys=True, separators=(',', ':'))
            strategy_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]
            
            # Build index entry following trace-updates.md structure
            entry = {
                'strategy_id': strategy_id,
                'strategy_hash': strategy_hash,
                'strategy_type': metadata.get('strategy_type', 'unknown'),
                'symbol': metadata.get('symbol', ''),
                'timeframe': metadata.get('timeframe', ''),
                'constraints': metadata.get('full_config', {}).get('constraints'),
            }
            
            # Add parameters as direct columns (not prefixed) per trace-updates.md
            # Common strategy parameters
            if 'period' in params:
                entry['period'] = params['period']
            if 'std_dev' in params:
                entry['std_dev'] = params['std_dev']
            if 'fast_period' in params:
                entry['fast_period'] = params['fast_period']
            if 'slow_period' in params:
                entry['slow_period'] = params['slow_period']
            if 'multiplier' in params:
                entry['multiplier'] = params['multiplier']
            if 'exit_threshold' in params:
                entry['exit_threshold'] = params['exit_threshold']
            
            # Add any other non-internal parameters
            for param_name, param_value in params.items():
                if not param_name.startswith('_') and param_name not in entry:
                    entry[param_name] = param_value
            
            # Add trace file path - must match actual storage structure
            # Files are stored in traces/signals/STRATEGY_TYPE/COMPONENT_ID.parquet
            trace_path = f"traces/signals/{metadata.get('strategy_type', 'unknown')}/{strategy_id}.parquet"
            entry['trace_path'] = trace_path
            
            index_data.append(entry)
        
        if index_data:
            # Create DataFrame and save as parquet
            index_df = pd.DataFrame(index_data)
            
            # Log what we're creating for debugging
            logger.info(f"Creating strategy index with {len(index_data)} strategies")
            logger.info(f"Columns: {list(index_df.columns)}")
            logger.info(f"Unique hashes: {index_df['strategy_hash'].nunique()}")
            
            index_path = self._workspace_path / 'strategy_index.parquet'
            index_df.to_parquet(index_path, engine='pyarrow', index=False)
            logger.info(f"Saved strategy index to {index_path}")
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