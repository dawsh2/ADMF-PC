"""
Multi-Strategy Tracer

A single tracer that attaches to the root event bus and manages separate
sparse storage instances for each strategy and classifier. This simplifies
the architecture by removing tracing logic from containers.

Compliant with COMPREHENSIVE_SQL_ANALYTICS_ARCHITECTURE.md:
- Hierarchical storage: signals/[type]/[id].parquet
- Sparse signal format: only stores changes
- SQL catalog ready with metadata tracking
"""

from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import logging
from datetime import datetime
import json
import hashlib

from ..types import Event, EventType
from ..protocols import EventObserverProtocol
from ..storage.temporal_sparse_storage import TemporalSparseStorage
from ..storage.dense_event_storage import DenseEventStorage
from .strategy_metadata_extractor import update_metadata_with_recursive_strategies

logger = logging.getLogger(__name__)


class MultiStrategyTracer(EventObserverProtocol):
    """
    Unified tracer for all strategies and classifiers.
    
    Manages separate sparse storage instances for each component,
    organizing them hierarchically according to SQL analytics architecture.
    """
    
    def __init__(self, 
                 workspace_path: str,
                 workflow_id: str,
                 managed_strategies: Optional[List[str]] = None,
                 managed_classifiers: Optional[List[str]] = None,
                 data_source_config: Optional[Dict[str, Any]] = None,
                 full_config: Optional[Dict[str, Any]] = None):
        """
        Initialize multi-strategy tracer.
        
        Args:
            workspace_path: Base directory for signal storage
            workflow_id: Workflow identifier for organization
            managed_strategies: Optional list of specific strategies to trace
            managed_classifiers: Optional list of specific classifiers to trace
            data_source_config: Configuration with data source info (symbols, timeframes, data_dir, etc.)
            full_config: Full configuration dict containing strategy definitions for metadata extraction
        """
        self._workspace_path = Path(workspace_path)
        self._workflow_id = workflow_id
        self._data_source_config = data_source_config or {}
        self._full_config = full_config or {}
        
        # If not specified, trace all
        self._managed_strategies = set(managed_strategies) if managed_strategies else None
        self._managed_classifiers = set(managed_classifiers) if managed_classifiers else None
        
        # Storage instances per component (sparse for signals)
        self._storages: Dict[str, TemporalSparseStorage] = {}
        
        # Dense storage for orders, fills, and positions
        self._dense_storages: Dict[str, DenseEventStorage] = {}
        
        # Current bar count (shared across all components) - 0-based to match source files
        self._current_bar_count = 0
        self._initialized_bar_count = False
        
        # Metrics
        self._total_signals = 0
        self._total_classifications = 0
        self._stored_changes = 0
        
        # Component metadata for SQL catalog
        self._component_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Create base traces directory - symbol_timeframe subdirs created on demand
        self._traces_dir = self._workspace_path / "traces"
        self._traces_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"MultiStrategyTracer initialized for workspace: {workspace_path}")
        if managed_strategies:
            logger.debug(f"Tracing strategies: {managed_strategies}")
        if managed_classifiers:
            logger.debug(f"Tracing classifiers: {managed_classifiers}")
    
    def on_event(self, event: Event) -> None:
        """Process events from the root event bus."""
        if event.event_type == EventType.BAR.value:
            # Use original bar index from event payload for consistent sparse storage
            has_payload = hasattr(event, 'payload')
            has_original_index = has_payload and event.payload and 'original_bar_index' in event.payload
            
            if has_original_index:
                self._current_bar_count = event.payload['original_bar_index']  # Keep 0-based indexing to match source
                logger.debug(f"📊 Bar count (original): {self._current_bar_count}")
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
                logger.debug(f"📊 Bar count (incremented): {self._current_bar_count}")
            
        elif event.event_type == EventType.SIGNAL.value:
            logger.debug(f"[TRACER] Received SIGNAL event from {event.payload.get('strategy_id', 'unknown')}")
            self._process_signal_event(event)
            
        elif event.event_type == EventType.CLASSIFICATION.value:
            logger.debug(f"MultiStrategyTracer received CLASSIFICATION event")
            self._process_classification_event(event)
            
        elif event.event_type == EventType.POSITION_OPEN.value:
            logger.debug(f"MultiStrategyTracer received POSITION_OPEN event")
            self._process_position_event(event)
            
        elif event.event_type == EventType.POSITION_CLOSE.value:
            logger.debug(f"MultiStrategyTracer received POSITION_CLOSE event")
            self._process_position_event(event)
            
        elif event.event_type == EventType.ORDER.value:
            logger.debug(f"MultiStrategyTracer received ORDER event")
            self._process_order_event(event)
            
        elif event.event_type == EventType.FILL.value:
            logger.debug(f"MultiStrategyTracer received FILL event")
            self._process_fill_event(event)
    
    def _process_signal_event(self, event: Event) -> None:
        """Process SIGNAL events from strategies."""
        payload = event.payload
        strategy_id = payload.get('strategy_id', '')
        
        logger.debug(f"[TRACER] Processing signal from {strategy_id}, payload keys: {list(payload.keys())}")
        
        # Check if we should trace this strategy
        if self._managed_strategies and strategy_id not in self._managed_strategies:
            # Also check if any managed strategy is a substring (for pattern matching)
            is_managed = any(ms in strategy_id for ms in self._managed_strategies)
            if not is_managed:
                logger.debug(f"Skipping unmanaged strategy: {strategy_id}")
                return
        
        self._total_signals += 1
        
        # Get or create storage for this strategy
        storage = self._get_or_create_storage('strategy', strategy_id, payload)
        
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
        
        # Extract price from metadata where strategies put it
        metadata = payload.get('metadata', {})
        price = metadata.get('price', payload.get('price', 0.0))
        
        was_change = storage.process_signal(
            symbol=symbol,
            direction=str(direction),
            strategy_id=strategy_id,
            timestamp=timestamp,
            price=price,
            bar_index=self._current_bar_count
        )
        
        if was_change:
            self._stored_changes += 1
            
        # Log compression ratio periodically
        if self._total_signals % 1000 == 0:
            ratio = (self._stored_changes / self._total_signals * 100) if self._total_signals > 0 else 0
            logger.debug(f"MultiStrategyTracer: {self._stored_changes}/{self._total_signals} "
                        f"signals stored ({ratio:.1f}% compression)")
    
    def _process_classification_event(self, event: Event) -> None:
        """Process CLASSIFICATION events from classifiers."""
        payload = event.payload
        classifier_id = payload.get('classifier_id', '')
        
        # Check if we should trace this classifier
        if self._managed_classifiers and classifier_id not in self._managed_classifiers:
            # Also check if any managed classifier is a substring
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
        
        # Classifications are already sparse (only published on change)
        was_change = storage.process_signal(
            symbol=symbol,
            direction=str(regime),  # Store regime as direction
            strategy_id=classifier_id,
            timestamp=timestamp,
            price=0.0,  # Classifications don't have prices
            bar_index=self._current_bar_count
        )
        
        if was_change:
            self._stored_changes += 1
    
    def _get_or_create_storage(self, component_type: str, component_id: str, 
                              payload: Dict[str, Any]) -> TemporalSparseStorage:
        """Get or create a storage instance for a component."""
        if component_id in self._storages:
            return self._storages[component_id]
        
        # Extract symbol and timeframe from payload or component_id
        symbol = payload.get('symbol', 'UNKNOWN')
        timeframe = payload.get('timeframe', '1m')  # Default to 1m
        
        # For backwards compatibility, extract symbol from component_id if not in payload
        if symbol == 'UNKNOWN':
            component_parts = component_id.split('_')
            if component_parts and component_parts[0] in ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'TSLA']:
                symbol = component_parts[0]
        
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
                # Grid search pattern: type is everything before 'grid'
                grid_idx = component_parts.index('grid')
                # Skip symbol if it's the first part
                start_idx = 1 if component_parts[0] == symbol else 0
                strategy_type = '_'.join(component_parts[start_idx:start_idx + grid_idx])
            else:
                # Extract from common patterns
                for pattern in ['momentum', 'ma_crossover', 'mean_reversion', 'breakout', 'rsi', 'macd', 
                               'regime', 'volatility', 'trend', 'market_state']:
                    if pattern in component_id:
                        strategy_type = pattern
                        break
                else:
                    strategy_type = 'unknown'
        
        # Create hierarchical directory structure: traces/STRATEGY_TYPE/
        # Simplified structure without symbol_timeframe subdirectory
        if component_type == 'strategy':
            base_dir = self._traces_dir / strategy_type
        else:
            base_dir = self._traces_dir / "classifiers" / strategy_type
        
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Get source data file path
        data_dir = self._data_source_config.get('data_dir', './data')
        source_file_path = f"{data_dir}/{symbol}_{timeframe}.csv"  # Standard naming convention
        data_source_type = self._data_source_config.get('data_source', 'csv')
        
        # Create storage with source metadata
        storage = TemporalSparseStorage(
            base_dir=str(base_dir),
            run_id=component_id,
            timeframe=timeframe,
            source_file_path=source_file_path,
            data_source_type=data_source_type
        )
        
        self._storages[component_id] = storage
        
        # Store metadata for SQL catalog
        # Get parameters from structured payload first, fallback to metadata
        parameters = payload.get('parameters', {})
        if not parameters and 'metadata' in payload:
            parameters = payload['metadata'].get('parameters', {})
        
        # Debug logging for parameter extraction
        if not parameters:
            logger.warning(f"[TRACER] No parameters found for {component_id}")
            logger.debug(f"[TRACER] Payload keys: {list(payload.keys())}")
            if 'metadata' in payload:
                logger.debug(f"[TRACER] Metadata keys: {list(payload['metadata'].keys())}")
        
        # For ensemble strategies, we need to look deeper for actual strategy parameters
        if strategy_type == 'ensemble' and not parameters:
            # Try to get from the metadata's composite_strategies
            metadata = payload.get('metadata', {})
            composite_strategies = metadata.get('composite_strategies', [])
            if composite_strategies:
                # For now, just log what we found
                logger.info(f"[TRACER] Ensemble strategy has {len(composite_strategies)} sub-strategies")
                # For single-strategy ensembles, extract the actual strategy parameters
                if len(composite_strategies) == 1 and isinstance(composite_strategies[0], dict):
                    first_strategy = composite_strategies[0]
                    # Use the sub-strategy's parameters directly
                    parameters = first_strategy.get('params', {})
                    # Also update the strategy_type to reflect the actual strategy
                    actual_strategy_type = first_strategy.get('type', 'ensemble')
                    if actual_strategy_type != 'ensemble':
                        strategy_type = actual_strategy_type
                        logger.info(f"[TRACER] Single-strategy ensemble, using actual type: {strategy_type}")
                else:
                    # Multi-strategy ensemble
                    parameters = {
                        'ensemble_type': 'composite',
                        'num_strategies': len(composite_strategies),
                        'sub_strategies': composite_strategies
                    }
                logger.info(f"[TRACER] Extracted ensemble parameters: {parameters}")
        
        # Get strategy hash from payload if available
        strategy_hash = payload.get('strategy_hash')
        
        self._component_metadata[component_id] = {
            'component_type': component_type,
            'strategy_type': strategy_type,
            'component_id': component_id,
            'parameters': parameters,
            'strategy_hash': strategy_hash,  # Store pre-computed hash if available
            'created_at': datetime.now().isoformat()
        }
        
        # Debug: Log when we successfully extract parameters
        if parameters and component_type == 'strategy':
            logger.info(f"[TRACER] Stored parameters for {component_id}: {list(parameters.keys())}")
        
        logger.debug(f"Created storage for {component_type} {component_id} in {base_dir}")
        
        return storage
    
    def finalize(self) -> Dict[str, Any]:
        """
        Finalize tracing and save all signals.
        
        Returns metadata about the traced components and their storage locations.
        """
        logger.debug("Finalizing MultiStrategyTracer...")
        
        results = {
            'workflow_id': self._workflow_id,
            'workspace_path': str(self._workspace_path),  # Add workspace path for analytics integration
            'total_bars': self._current_bar_count,
            'total_signals': self._total_signals,
            'total_classifications': self._total_classifications,
            'stored_changes': self._stored_changes,
            'compression_ratio': (self._stored_changes / (self._total_signals + self._total_classifications) * 100) 
                                if (self._total_signals + self._total_classifications) > 0 else 0,
            'components': {}
        }
        
        # Save dense event storages first (orders, fills, positions)
        for storage_key, dense_storage in self._dense_storages.items():
            filepath = dense_storage.save(f"{storage_key}.parquet")
            if filepath:
                logger.info(f"Saved dense storage {storage_key} to {filepath}")
        
        # Save each component's signals (sparse storage)
        for component_id, storage in self._storages.items():
            metadata = self._component_metadata.get(component_id, {})
            component_type = metadata.get('component_type', 'unknown')
            strategy_type = metadata.get('strategy_type', 'unknown')
            
            # Save with component_id as tag for unique filenames
            filepath = storage.save(tag=component_id)
            
            if filepath:
                # Calculate relative path for SQL catalog (relative to workspace root)
                rel_path = Path(filepath).relative_to(self._workspace_path)
                
                results['components'][component_id] = {
                    'component_type': component_type,
                    'strategy_type': strategy_type,
                    'signal_file_path': str(rel_path),
                    'total_bars': storage._bar_index,
                    'signal_changes': len(storage._changes),
                    'compression_ratio': len(storage._changes) / storage._bar_index if storage._bar_index > 0 else 0,
                    'signal_frequency': len(storage._changes) / storage._bar_index if storage._bar_index > 0 else 0,
                    'parameters': metadata.get('parameters', {}),
                    'created_at': metadata.get('created_at')
                }
                
                logger.debug(f"Saved {component_id}: {len(storage._changes)} changes to {rel_path}")
        
        # Add recursive strategy metadata if full config was provided
        if self._full_config:
            try:
                logger.info(f"[TRACER] Extracting metadata from config with keys: {list(self._full_config.keys())}")
                if 'strategy' in self._full_config:
                    logger.info(f"[TRACER] Strategy value: {self._full_config['strategy']}")
                if 'strategies' in self._full_config:
                    logger.info(f"[TRACER] Strategies value: {self._full_config['strategies']}")
                results = update_metadata_with_recursive_strategies(results, self._full_config)
                logger.debug("Added recursive strategy metadata to results")
            except Exception as e:
                logger.error(f"Failed to extract recursive strategy metadata: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Continue without recursive metadata rather than failing
        
        # Save metadata.json at workspace root for SQL catalog population
        metadata_path = self._workspace_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create strategy index (following trace-updates.md)
        self._create_strategy_index()
        
        logger.debug(f"MultiStrategyTracer finalized. Metadata saved to {metadata_path}")
        logger.debug(f"Overall compression: {results['compression_ratio']:.1f}%")
        
        return results
    
    def _create_strategy_index(self) -> None:
        """Create a queryable index of all strategies in this run.
        
        Following trace-updates.md specification:
        - Parameters stored as direct columns (period, std_dev, etc)
        - Unique strategy hash per configuration
        - Self-documenting with embedded metadata
        """
        import pandas as pd
        import hashlib
        
        index_data = []
        
        # Extract from component metadata stored during execution
        for component_id, component_data in self._component_metadata.items():
            if component_data.get('component_type') != 'strategy':
                continue
                
            # Extract parameters
            params = component_data.get('parameters', {})
            strategy_type = component_data.get('strategy_type', 'unknown')
            
            # Use pre-computed hash if available, otherwise generate it
            strategy_hash = component_data.get('strategy_hash')
            if not strategy_hash:
                # Compute unique hash based on actual parameters
                hash_config = {
                    'type': strategy_type,
                    'parameters': {k: v for k, v in params.items() if not k.startswith('_')}
                }
                config_str = json.dumps(hash_config, sort_keys=True, separators=(',', ':'))
                strategy_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]
                logger.debug(f"Generated hash for {component_id}: {strategy_hash}")
            
            # Build index entry following trace-updates.md structure
            entry = {
                'strategy_id': component_id,
                'strategy_hash': strategy_hash,
                'strategy_type': strategy_type,
                'symbol': component_id.split('_')[0] if '_' in component_id else 'UNKNOWN',
                'timeframe': '5m',  # Default, could be extracted from metadata
                'constraints': None,  # Could be extracted from config
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
            
            # Add trace file path based on where we saved it
            # Files are stored in traces/signals/STRATEGY_TYPE/COMPONENT_ID.parquet
            trace_path = f"traces/signals/{strategy_type}/{component_id}.parquet"
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
    
    def _process_position_event(self, event: Event) -> None:
        """Process POSITION_OPEN and POSITION_CLOSE events from portfolio."""
        payload = event.payload
        symbol = payload.get('symbol', 'UNKNOWN')
        timestamp = event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp)
        
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
        
        logger.info(f"Stored {event.event_type} event for {symbol}")
    
    def _process_order_event(self, event: Event) -> None:
        """Process ORDER events from portfolio."""
        payload = event.payload
        symbol = payload.get('symbol', 'UNKNOWN')
        timestamp = event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp)
        
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
        logger.info(f"Stored ORDER event for {symbol} with id {order_id}")
    
    def _process_fill_event(self, event: Event) -> None:
        """Process FILL events from execution engine."""
        payload = event.payload
        symbol = payload.get('symbol', 'UNKNOWN')
        timestamp = event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp)
        
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
        logger.info(f"Stored FILL event for {symbol} with id {fill_id}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current tracing metrics."""
        return {
            'current_bar': self._current_bar_count,
            'total_signals': self._total_signals,
            'total_classifications': self._total_classifications,
            'stored_changes': self._stored_changes,
            'active_components': len(self._storages),
            'compression_ratio': (self._stored_changes / (self._total_signals + self._total_classifications) * 100) 
                                if (self._total_signals + self._total_classifications) > 0 else 0
        }
    
    # Implement EventObserverProtocol abstract methods
    def on_publish(self, event: Event) -> None:
        """Called when an event is published. Process the event here instead of on_event."""
        self.on_event(event)
    
    def on_delivered(self, event: Event, handler_count: int) -> None:
        """Called after event is delivered to handlers. We don't need this."""
        pass
    
    def on_error(self, event: Event, error: Exception) -> None:
        """Called when an error occurs during event processing."""
        logger.error(f"Error processing event {event.event_type}: {error}")