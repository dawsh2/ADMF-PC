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

from ..types import Event, EventType
from ..protocols import EventObserverProtocol
from ..storage.streaming_sparse_storage import StreamingSparseStorage

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
                 write_on_changes: int = 0):
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
        """
        self._workspace_path = Path(workspace_path)
        self._workflow_id = workflow_id
        self._data_source_config = data_source_config or {}
        self._write_interval = write_interval
        self._write_on_changes = write_on_changes
        
        # If not specified, trace all
        self._managed_strategies = set(managed_strategies) if managed_strategies else None
        self._managed_classifiers = set(managed_classifiers) if managed_classifiers else None
        
        # Storage instances per component
        self._storages: Dict[str, StreamingSparseStorage] = {}
        
        # Current bar count
        self._current_bar_count = 0
        
        # Metrics
        self._total_signals = 0
        self._total_classifications = 0
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
        if event.event_type == EventType.BAR.value:
            # Use original bar index from event payload for consistent sparse storage
            if hasattr(event, 'payload') and 'original_bar_index' in event.payload:
                self._current_bar_count = event.payload['original_bar_index'] + 1  # Convert to 1-based
            else:
                # Fallback to incrementing for backward compatibility
                self._current_bar_count += 1
            
            # Log progress periodically
            if self._current_bar_count % 100 == 0:
                logger.info(f"Processed {self._current_bar_count} bars, "
                           f"{self._stored_changes} signal changes stored")
            
        elif event.event_type == EventType.SIGNAL.value:
            self._process_signal_event(event)
            
        elif event.event_type == EventType.CLASSIFICATION.value:
            self._process_classification_event(event)
    
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
        
        # Process the signal
        direction = payload.get('direction', 'flat')
        symbol = payload.get('symbol', 'UNKNOWN')
        timestamp = event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp)
        price = payload.get('price', 0.0)
        
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
    
    def _get_or_create_storage(self, component_type: str, component_id: str, 
                              payload: Dict[str, Any]) -> StreamingSparseStorage:
        """Get or create a storage instance for a component."""
        if component_id in self._storages:
            return self._storages[component_id]
        
        # Extract symbol and timeframe
        symbol = payload.get('symbol', 'UNKNOWN')
        timeframe = payload.get('timeframe', '1m')
        
        # Extract strategy/classifier type for directory organization
        component_parts = component_id.split('_')
        if 'grid' in component_id:
            # Find the grid pattern (e.g., ma_crossover_grid, rsi_grid)
            grid_idx = component_parts.index('grid')
            start_idx = 1 if component_parts[0] == symbol else 0
            strategy_type = '_'.join(component_parts[start_idx:grid_idx + 1])  # Include 'grid'
        else:
            start_idx = 1 if component_parts[0] == symbol else 0
            strategy_type = component_parts[start_idx] if start_idx < len(component_parts) else 'unknown'
        
        # Create subdirectory structure: traces/SYMBOL_TIMEFRAME/signals|classifiers/STRATEGY_TYPE/
        symbol_timeframe_dir = self._traces_dir / f"{symbol}_{timeframe}"
        if component_type == 'strategy':
            component_dir = symbol_timeframe_dir / 'signals' / strategy_type
        else:
            component_dir = symbol_timeframe_dir / 'classifiers' / strategy_type
        
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
            self._component_metadata[component_id] = {
                'type': component_type,
                'strategy_type': strategy_type,
                'symbol': symbol,
                'timeframe': timeframe,
                'parameters': payload.get('parameters', {})
            }
        
        logger.info(f"Created streaming storage for {component_type} {component_id} at {component_dir}")
        
        return storage
    
    def finalize(self) -> Dict[str, Any]:
        """Finalize all storages and save metadata."""
        logger.info(f"Finalizing StreamingMultiStrategyTracer...")
        
        # Finalize all storages
        for component_id, storage in self._storages.items():
            try:
                storage.finalize()
            except Exception as e:
                logger.error(f"Error finalizing storage for {component_id}: {e}")
        
        # Save workspace metadata
        metadata = {
            'workflow_id': self._workflow_id,
            'data_source': self._data_source_config,
            'total_bars': self._current_bar_count,
            'total_signals': self._total_signals,
            'total_classifications': self._total_classifications,
            'stored_changes': self._stored_changes,
            'compression_ratio': self._stored_changes / self._total_signals if self._total_signals > 0 else 0,
            'components': self._component_metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = self._workspace_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Workspace finalized: {self._workspace_path}")
        logger.info(f"Total signals: {self._total_signals}, "
                   f"Changes stored: {self._stored_changes} "
                   f"({self._stored_changes/self._total_signals*100:.1f}% compression)")
        
        return metadata
    
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