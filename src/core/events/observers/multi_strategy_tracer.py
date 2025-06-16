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

from ..types import Event, EventType
from ..protocols import EventObserverProtocol
from ..storage.temporal_sparse_storage import TemporalSparseStorage

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
                 data_source_config: Optional[Dict[str, Any]] = None):
        """
        Initialize multi-strategy tracer.
        
        Args:
            workspace_path: Base directory for signal storage
            workflow_id: Workflow identifier for organization
            managed_strategies: Optional list of specific strategies to trace
            managed_classifiers: Optional list of specific classifiers to trace
            data_source_config: Configuration with data source info (symbols, timeframes, data_dir, etc.)
        """
        self._workspace_path = Path(workspace_path)
        self._workflow_id = workflow_id
        self._data_source_config = data_source_config or {}
        
        # If not specified, trace all
        self._managed_strategies = set(managed_strategies) if managed_strategies else None
        self._managed_classifiers = set(managed_classifiers) if managed_classifiers else None
        
        # Storage instances per component
        self._storages: Dict[str, TemporalSparseStorage] = {}
        
        # Current bar count (shared across all components)
        self._current_bar_count = 0
        
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
            if hasattr(event, 'payload') and 'original_bar_index' in event.payload:
                self._current_bar_count = event.payload['original_bar_index'] + 1  # Convert to 1-based
                logger.debug(f"Bar count (original): {self._current_bar_count}")
            else:
                # Fallback to incrementing for backward compatibility
                self._current_bar_count += 1
                logger.debug(f"Bar count (incremented): {self._current_bar_count}")
            
        elif event.event_type == EventType.SIGNAL.value:
            logger.debug(f"MultiStrategyTracer received SIGNAL event")
            self._process_signal_event(event)
            
        elif event.event_type == EventType.CLASSIFICATION.value:
            logger.debug(f"MultiStrategyTracer received CLASSIFICATION event")
            self._process_classification_event(event)
    
    def _process_signal_event(self, event: Event) -> None:
        """Process SIGNAL events from strategies."""
        payload = event.payload
        strategy_id = payload.get('strategy_id', '')
        
        logger.debug(f"Processing signal from {strategy_id}")
        
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
        
        # Extract strategy/classifier type for directory organization
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
        
        # Create hierarchical directory structure: traces/SYMBOL_TIMEFRAME/{signals,classifiers}/STRATEGY_TYPE/
        symbol_timeframe_dir = self._traces_dir / f"{symbol}_{timeframe}"
        
        if component_type == 'strategy':
            base_dir = symbol_timeframe_dir / "signals" / strategy_type
        else:
            base_dir = symbol_timeframe_dir / "classifiers" / strategy_type
        
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
        self._component_metadata[component_id] = {
            'component_type': component_type,
            'strategy_type': strategy_type,
            'component_id': component_id,
            'parameters': payload.get('metadata', {}).get('parameters', {}),
            'created_at': datetime.now().isoformat()
        }
        
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
        
        # Save each component's signals
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
        
        # Save metadata.json at workspace root for SQL catalog population
        metadata_path = self._workspace_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.debug(f"MultiStrategyTracer finalized. Metadata saved to {metadata_path}")
        logger.debug(f"Overall compression: {results['compression_ratio']:.1f}%")
        
        return results
    
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