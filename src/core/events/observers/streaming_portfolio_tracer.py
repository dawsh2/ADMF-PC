"""
Streaming Portfolio Tracer with Periodic Disk Writes

Enhanced version of HierarchicalPortfolioTracer that writes to disk periodically
to prevent memory accumulation during long runs.
"""

from typing import Optional, Dict, Any, List, Set
import logging
from datetime import datetime
from pathlib import Path

from ..types import Event, EventType
from ..protocols import EventObserverProtocol
from ..storage.simple_parquet_storage import MinimalHierarchicalStorage

logger = logging.getLogger(__name__)


class StreamingPortfolioTracer(EventObserverProtocol):
    """
    Portfolio tracer that streams signals to disk periodically.
    
    Key improvements:
    1. Periodic writes to disk (configurable interval)
    2. Minimal state tracking (only last signal per strategy)
    3. Automatic buffer clearing after writes
    """
    
    def __init__(self, 
                 container_id: str,
                 workflow_id: str,
                 managed_strategies: List[str],
                 managed_classifiers: Optional[List[str]] = None,
                 storage_config: Optional[Dict[str, Any]] = None,
                 portfolio_container: Optional[Any] = None):
        """Initialize streaming tracer."""
        self.container_id = container_id
        self.workflow_id = workflow_id
        self.managed_strategies = set(managed_strategies)
        self.managed_classifiers = set(managed_classifiers or [])
        self.portfolio_container = portfolio_container
        
        # Configure storage
        config = storage_config or {}
        base_dir = config.get('base_dir', './analytics_storage')
        self.write_interval = config.get('write_interval', 500)  # Write every 500 bars
        self.write_on_change_count = config.get('write_on_changes', 100)  # Or every 100 changes
        
        # Use hierarchical storage
        self.storage = MinimalHierarchicalStorage(base_dir=base_dir)
        
        # Minimal buffers - cleared after each write
        self.signal_buffers: Dict[str, List[Dict]] = {}
        self.classifier_buffers: Dict[str, List[Dict]] = {}
        
        # Only track last signal/regime per source (not full history)
        self.last_signals: Dict[str, Any] = {}  # strategy_id -> last signal value
        self.last_regimes: Dict[str, Any] = {}  # classifier_id -> last regime
        
        # Metadata tracking (persisted across writes)
        self.strategy_metadata: Dict[str, Dict[str, Any]] = {}
        self.classifier_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Write tracking
        self.bar_index = 0
        self.total_changes = 0
        self.changes_since_write = 0
        self.last_write_bar = 0
        self.write_count = 0
        
        # File tracking for append mode
        self.file_counters: Dict[str, int] = {}
        
        logger.info(f"StreamingPortfolioTracer initialized for {container_id} "
                   f"with write_interval={self.write_interval} bars")
    
    def on_event(self, event: Event) -> None:
        """Process event - store signal and classification changes."""
        if event.event_type == EventType.SIGNAL.value:
            self._process_signal_event(event)
        elif event.event_type == EventType.CLASSIFICATION.value:
            self._process_classification_event(event)
        elif event.event_type == EventType.MARKET_DATA.value:
            # Increment bar counter and check if we should write
            self.bar_index += 1
            self._check_write_conditions()
    
    def _process_signal_event(self, event: Event) -> None:
        """Process SIGNAL events from strategies."""
        payload = event.payload
        strategy_id = payload.get('strategy_id', '')
        
        # Check if this signal is from a managed strategy
        is_managed = any(strategy_name in strategy_id for strategy_name in self.managed_strategies)
        if not is_managed:
            return
        
        # Extract signal details
        symbol = payload.get('symbol', 'UNKNOWN')
        direction = payload.get('direction', 'flat')
        timestamp = event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp)
        price = payload.get('price', 0.0)
        
        # Convert direction to numeric value
        signal_value = 1 if direction == 'long' else (-1 if direction == 'short' else 0)
        
        # Check if this is a change (comparing only to last stored signal)
        last_value = self.last_signals.get(strategy_id)
        
        if last_value is None or last_value != signal_value:
            # Record the change
            change = {
                'idx': self.bar_index,
                'ts': timestamp,
                'sym': symbol,
                'val': signal_value,
                'strat': strategy_id,
                'px': price
            }
            
            # Add to buffer
            if strategy_id not in self.signal_buffers:
                self.signal_buffers[strategy_id] = []
            self.signal_buffers[strategy_id].append(change)
            
            # Update last signal
            self.last_signals[strategy_id] = signal_value
            
            self.total_changes += 1
            self.changes_since_write += 1
            
            # Store strategy metadata once
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
        last_regime = self.last_regimes.get(classifier_id)
        
        if last_regime is None or last_regime != regime:
            # Record the change
            change = {
                'idx': self.bar_index,
                'ts': timestamp,
                'regime': regime,
                'confidence': confidence,
                'previous_regime': last_regime
            }
            
            # Add to buffer
            if classifier_id not in self.classifier_buffers:
                self.classifier_buffers[classifier_id] = []
            self.classifier_buffers[classifier_id].append(change)
            
            # Update last regime
            self.last_regimes[classifier_id] = regime
            
            self.changes_since_write += 1
            
            # Store classifier metadata once
            if 'parameters' in payload and classifier_id not in self.classifier_metadata:
                self.classifier_metadata[classifier_id] = {
                    'params': payload['parameters'],
                    'type': payload.get('classifier_type', 'unknown')
                }
    
    def _check_write_conditions(self) -> None:
        """Check if we should write to disk based on conditions."""
        bars_since_write = self.bar_index - self.last_write_bar
        
        should_write = (
            # Write every N bars
            bars_since_write >= self.write_interval or
            # Write every N changes
            self.changes_since_write >= self.write_on_change_count
        )
        
        if should_write and self.changes_since_write > 0:
            self._write_buffers_to_disk()
    
    def _write_buffers_to_disk(self) -> None:
        """Write current buffers to disk and clear them."""
        logger.debug(f"Writing buffers to disk at bar {self.bar_index} "
                    f"({self.changes_since_write} changes)")
        
        # Save signal data
        for strategy_id, changes in self.signal_buffers.items():
            if not changes:
                continue
            
            # Extract strategy name from ID
            strategy_name = strategy_id.split('_', 1)[-1] if '_' in strategy_id else strategy_id
            
            # Get metadata
            meta = self.strategy_metadata.get(strategy_id, {})
            
            # Create unique file name for append mode
            file_key = f"signal_{strategy_id}"
            self.file_counters[file_key] = self.file_counters.get(file_key, 0) + 1
            
            try:
                storage_meta = self.storage.store_signal_data(
                    signal_changes=changes,
                    strategy_name=f"{strategy_name}_part{self.file_counters[file_key]:03d}",
                    parameters=meta.get('params', {}),
                    metadata={
                        'bars_start': self.last_write_bar,
                        'bars_end': self.bar_index,
                        'total_bars': self.bar_index,
                        'symbol': meta.get('symbol', 'UNKNOWN'),
                        'timeframe': meta.get('timeframe', '1m'),
                        'workflow_id': self.workflow_id,
                        'container_id': self.container_id,
                        'strategy_type': meta.get('type'),
                        'part_number': self.file_counters[file_key]
                    }
                )
                
                logger.debug(f"Wrote {len(changes)} signal changes for {strategy_name}")
                
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
            
            # Create unique file name
            file_key = f"classifier_{classifier_id}"
            self.file_counters[file_key] = self.file_counters.get(file_key, 0) + 1
            
            try:
                storage_meta = self.storage.store_classifier_data(
                    regime_changes=changes,
                    classifier_name=f"{classifier_name}_part{self.file_counters[file_key]:03d}",
                    parameters=meta.get('params', {}),
                    metadata={
                        'bars_start': self.last_write_bar,
                        'bars_end': self.bar_index,
                        'total_bars': self.bar_index,
                        'workflow_id': self.workflow_id,
                        'container_id': self.container_id,
                        'classifier_type': meta.get('type'),
                        'part_number': self.file_counters[file_key]
                    }
                )
                
                logger.debug(f"Wrote {len(changes)} regime changes for {classifier_name}")
                
            except Exception as e:
                logger.error(f"Failed to save classifier data for {classifier_id}: {e}")
        
        # Clear buffers to free memory
        self.signal_buffers.clear()
        self.classifier_buffers.clear()
        
        # Update tracking
        self.last_write_bar = self.bar_index
        self.changes_since_write = 0
        self.write_count += 1
        
        logger.info(f"Completed write #{self.write_count} at bar {self.bar_index}")
    
    def save(self) -> Dict[str, List[str]]:
        """
        Final save - write any remaining buffered data.
        
        Returns:
            Dictionary with paths to saved files
        """
        # Write any remaining data
        if self.changes_since_write > 0:
            self._write_buffers_to_disk()
        
        # Log final summary
        logger.info(f"StreamingPortfolioTracer final summary for {self.container_id}:")
        logger.info(f"  Total bars: {self.bar_index}")
        logger.info(f"  Total changes: {self.total_changes}")
        logger.info(f"  Total writes: {self.write_count}")
        logger.info(f"  Average changes per write: {self.total_changes/self.write_count:.1f}" if self.write_count > 0 else "  No writes")
        
        # Return empty dict as files were already written incrementally
        return {'signals': [], 'classifiers': []}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of traced data."""
        return {
            'container_id': self.container_id,
            'workflow_id': self.workflow_id,
            'total_bars': self.bar_index,
            'total_changes': self.total_changes,
            'write_count': self.write_count,
            'strategies_tracked': len(self.last_signals),
            'classifiers_tracked': len(self.last_regimes),
            'current_buffer_size': sum(len(b) for b in self.signal_buffers.values())
        }
    
    # Implement required protocol methods
    def on_publish(self, event: Event) -> None:
        """Called when event is published."""
        self.on_event(event)
    
    def on_delivered(self, event: Event, handler: Any) -> None:
        """Called when event is delivered to handler."""
        pass
    
    def on_error(self, event: Event, handler: Any, error: Exception) -> None:
        """Called when handler raises error."""
        logger.error(f"Handler error for {event.event_type}: {error}")