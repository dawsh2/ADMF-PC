"""
Hierarchical Portfolio Tracer

Enhanced sparse portfolio tracer that uses hierarchical Parquet storage
instead of flat JSON files in tmp/ directory.
"""

from typing import Optional, Dict, Any, List, Set
import logging
from datetime import datetime
from pathlib import Path

from ..types import Event, EventType
from ..protocols import EventObserverProtocol
from ..storage.simple_parquet_storage import MinimalHierarchicalStorage

logger = logging.getLogger(__name__)


class HierarchicalPortfolioTracer(EventObserverProtocol):
    """
    Portfolio tracer that stores signals and classifiers in hierarchical Parquet structure.
    
    Instead of creating messy JSON files in tmp/, creates organized Parquet files:
    - signals/momentum/mom_10_20_30_a1b2c3.parquet
    - classifiers/regime/hmm_3state_f6g7h8.parquet
    """
    
    def __init__(self, 
                 container_id: str,
                 workflow_id: str,
                 managed_strategies: List[str],
                 managed_classifiers: Optional[List[str]] = None,
                 storage_config: Optional[Dict[str, Any]] = None,
                 portfolio_container: Optional[Any] = None):
        """Initialize hierarchical tracer."""
        self.container_id = container_id
        self.workflow_id = workflow_id
        self.managed_strategies = set(managed_strategies)
        self.managed_classifiers = set(managed_classifiers or [])
        self.portfolio_container = portfolio_container
        
        # Configure storage
        config = storage_config or {}
        base_dir = config.get('base_dir', './analytics_storage')
        
        # Use hierarchical storage
        self.storage = MinimalHierarchicalStorage(base_dir=base_dir)
        
        # Track signals and classifiers by source
        self.signal_buffers: Dict[str, List[Dict]] = {}
        self.classifier_buffers: Dict[str, List[Dict]] = {}
        
        # Track current states to detect changes
        self.current_signal_states: Dict[str, Any] = {}
        self.current_classifier_states: Dict[str, Any] = {}
        
        # Metadata tracking
        self.strategy_metadata: Dict[str, Dict[str, Any]] = {}
        self.classifier_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Bar counting
        self.bar_index = 0
        self.total_signals = 0
        self.stored_changes = 0
        
        logger.info(f"HierarchicalPortfolioTracer initialized for {container_id} "
                   f"with storage at {base_dir}")
    
    def on_event(self, event: Event) -> None:
        """Process event - store signal and classification changes."""
        if event.event_type == EventType.SIGNAL.value:
            self._process_signal_event(event)
        elif event.event_type == EventType.CLASSIFICATION.value:
            self._process_classification_event(event)
        elif event.event_type == EventType.MARKET_DATA.value:
            # Increment bar counter on market data
            self.bar_index += 1
    
    def _process_signal_event(self, event: Event) -> None:
        """Process SIGNAL events from strategies."""
        payload = event.payload
        strategy_id = payload.get('strategy_id', '')
        
        # Check if this signal is from a managed strategy
        is_managed = any(strategy_name in strategy_id for strategy_name in self.managed_strategies)
        if not is_managed:
            return
        
        self.total_signals += 1
        
        # Extract signal details
        symbol = payload.get('symbol', 'UNKNOWN')
        direction = payload.get('direction', 'flat')
        timestamp = event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp)
        price = payload.get('price', 0.0)
        
        # Convert direction to numeric value
        if direction == 'long':
            signal_value = 1
        elif direction == 'short':
            signal_value = -1
        else:
            signal_value = 0
        
        # Check if this is a change
        state_key = f"{symbol}_{strategy_id}"
        current = self.current_signal_states.get(state_key)
        
        is_change = current is None or current['value'] != signal_value
        
        if is_change:
            # Record the change
            change = {
                'idx': self.bar_index,
                'ts': timestamp,
                'sym': symbol,
                'val': signal_value,
                'strat': strategy_id,
                'px': price
            }
            
            # Add to buffer for this strategy
            if strategy_id not in self.signal_buffers:
                self.signal_buffers[strategy_id] = []
            self.signal_buffers[strategy_id].append(change)
            
            # Update current state
            self.current_signal_states[state_key] = {
                'value': signal_value,
                'bar_index': self.bar_index,
                'timestamp': timestamp
            }
            
            self.stored_changes += 1
            
            # Store strategy metadata if available
            if 'parameters' in payload and strategy_id not in self.strategy_metadata:
                self.strategy_metadata[strategy_id] = {
                    'params': payload['parameters'],
                    'type': payload.get('strategy_type', 'unknown'),
                    'symbol': symbol,
                    'timeframe': payload.get('timeframe', '1m')
                }
    
    def _process_classification_event(self, event: Event) -> None:
        """Process CLASSIFICATION events from classifiers."""
        payload = event.payload
        classifier_id = payload.get('classifier_id', '')
        
        # Check if this is from a managed classifier
        is_managed = any(classifier_name in classifier_id for classifier_name in self.managed_classifiers)
        if not is_managed:
            return
        
        # Extract classification details
        regime = payload.get('regime', payload.get('state', 'UNKNOWN'))
        confidence = payload.get('confidence', 1.0)
        timestamp = event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp)
        
        # Check if this is a change
        state_key = classifier_id
        current = self.current_classifier_states.get(state_key)
        
        is_change = current is None or current['regime'] != regime
        
        if is_change:
            # Record the change
            change = {
                'idx': self.bar_index,
                'ts': timestamp,
                'regime': regime,
                'confidence': confidence,
                'previous_regime': current['regime'] if current else None
            }
            
            # Add to buffer
            if classifier_id not in self.classifier_buffers:
                self.classifier_buffers[classifier_id] = []
            self.classifier_buffers[classifier_id].append(change)
            
            # Update current state
            self.current_classifier_states[state_key] = {
                'regime': regime,
                'bar_index': self.bar_index,
                'timestamp': timestamp
            }
            
            # Store classifier metadata if available
            if 'parameters' in payload and classifier_id not in self.classifier_metadata:
                self.classifier_metadata[classifier_id] = {
                    'params': payload['parameters'],
                    'type': payload.get('classifier_type', 'unknown')
                }
    
    def save(self) -> Dict[str, List[str]]:
        """
        Save all buffered data to hierarchical Parquet storage.
        
        Returns:
            Dictionary with paths to saved files
        """
        saved_files = {
            'signals': [],
            'classifiers': []
        }
        
        # Save signal data
        for strategy_id, changes in self.signal_buffers.items():
            if not changes:
                continue
            
            # Extract strategy name from ID
            strategy_name = strategy_id.split('_', 1)[-1] if '_' in strategy_id else strategy_id
            
            # Get metadata
            meta = self.strategy_metadata.get(strategy_id, {})
            
            try:
                storage_meta = self.storage.store_signal_data(
                    signal_changes=changes,
                    strategy_name=strategy_name,
                    parameters=meta.get('params', {}),
                    metadata={
                        'total_bars': self.bar_index,
                        'symbol': meta.get('symbol', 'UNKNOWN'),
                        'timeframe': meta.get('timeframe', '1m'),
                        'workflow_id': self.workflow_id,
                        'container_id': self.container_id,
                        'strategy_type': meta.get('type')
                    }
                )
                saved_files['signals'].append(storage_meta.file_path)
                
                logger.info(f"Saved {len(changes)} signal changes for {strategy_name} "
                           f"to {storage_meta.file_path}")
                
            except Exception as e:
                logger.error(f"Failed to save signals for {strategy_id}: {e}")
        
        # Save classifier data
        for classifier_id, changes in self.classifier_buffers.items():
            if not changes:
                continue
            
            # Extract classifier name
            classifier_name = classifier_id.split('_', 1)[-1] if '_' in classifier_id else classifier_id
            
            # Get metadata
            meta = self.classifier_metadata.get(classifier_id, {})
            
            try:
                storage_meta = self.storage.store_classifier_data(
                    regime_changes=changes,
                    classifier_name=classifier_name,
                    parameters=meta.get('params', {}),
                    metadata={
                        'total_bars': self.bar_index,
                        'workflow_id': self.workflow_id,
                        'container_id': self.container_id,
                        'classifier_type': meta.get('type')
                    }
                )
                saved_files['classifiers'].append(storage_meta.file_path)
                
                logger.info(f"Saved {len(changes)} regime changes for {classifier_name} "
                           f"to {storage_meta.file_path}")
                
            except Exception as e:
                logger.error(f"Failed to save classifier data for {classifier_id}: {e}")
        
        # Log summary
        logger.info(f"Storage summary for {self.container_id}:")
        logger.info(f"  Total bars: {self.bar_index}")
        logger.info(f"  Total signals: {self.total_signals}")
        logger.info(f"  Stored changes: {self.stored_changes}")
        logger.info(f"  Compression: {self.stored_changes/self.total_signals*100:.1f}%" if self.total_signals > 0 else "  No signals")
        logger.info(f"  Saved {len(saved_files['signals'])} signal files, "
                   f"{len(saved_files['classifiers'])} classifier files")
        
        return saved_files
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of traced data."""
        return {
            'container_id': self.container_id,
            'workflow_id': self.workflow_id,
            'total_bars': self.bar_index,
            'total_signals': self.total_signals,
            'stored_changes': self.stored_changes,
            'compression_ratio': self.stored_changes / self.total_signals if self.total_signals > 0 else 0,
            'strategies_tracked': len(self.signal_buffers),
            'classifiers_tracked': len(self.classifier_buffers),
            'signal_changes_by_strategy': {k: len(v) for k, v in self.signal_buffers.items()},
            'regime_changes_by_classifier': {k: len(v) for k, v in self.classifier_buffers.items()}
        }
    
    # Required EventObserverProtocol methods
    def on_publish(self, event: Event) -> None:
        """Called when event is published."""
        self.on_event(event)
    
    def on_delivered(self, event: Event, handler_count: int) -> None:
        """Called after event delivery."""
        pass
    
    def on_error(self, event: Event, error: Exception, handler: Any = None) -> None:
        """Called on event error."""
        logger.error(f"Hierarchical tracer error for event {event.event_type}: {error}")