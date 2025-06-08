"""
Signal streaming component - compose into existing container types.

This component streams signals from storage, similar to how DataStreamer streams bars.
It supports regime filtering and boundary-aware replay for accurate backtesting.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime

from ...events import Event, EventType
from ...storage.signals import SignalStorageManager
from ..protocols import ContainerComponent


logger = logging.getLogger(__name__)


@dataclass
class BoundaryAwareReplay:
    """
    Handles signals that cross regime boundaries.
    
    Ensures that positions opened in a target regime are properly closed
    even if the regime changes before the exit signal.
    """
    target_regime: Optional[str] = None  # None means no filtering
    in_position: bool = False
    position_open_regime: Optional[str] = None
    position_open_bar: Optional[int] = None
    
    def should_emit_signal(self, signal_idx: int, signal_value: float, 
                          current_regime: Optional[str]) -> bool:
        """Determine if signal should be emitted based on regime rules."""
        # If no target regime specified, emit all signals
        if not self.target_regime:
            return True
        
        # Opening position - must be in target regime
        if signal_value != 0 and not self.in_position:
            if current_regime == self.target_regime:
                self.in_position = True
                self.position_open_regime = current_regime
                self.position_open_bar = signal_idx
                logger.debug(f"Opening position in {current_regime} regime at bar {signal_idx}")
                return True
            return False
        
        # Closing position - ALWAYS emit if we have open position
        elif signal_value == 0 and self.in_position:
            self.in_position = False
            if current_regime != self.position_open_regime:
                logger.info(f"Boundary trade: opened in {self.position_open_regime}, "
                           f"closed in {current_regime} (bars {self.position_open_bar}-{signal_idx})")
            return True
        
        return False


@dataclass
class SignalStreamerComponent(ContainerComponent):
    """
    Streams signals from storage, similar to how DataStreamer streams bars.
    
    Features:
    - Loads multiple signal files (from grid search)
    - Filters by strategy_id and regime
    - Supports sparse replay (only bars with signals)
    - Handles boundary trades correctly
    """
    
    # Configuration
    signal_storage_path: Path
    workflow_id: str
    
    # Filtering options
    strategy_filter: Optional[List[str]] = None
    regime_filter: Optional[str] = None
    sparse_replay: bool = True  # Only replay bars with signals
    
    # Storage manager
    storage_manager: Optional[SignalStorageManager] = None
    
    # Replay state
    current_index: int = 0
    bar_indices: List[int] = field(default_factory=list)
    signals_by_bar: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    
    # Boundary handling
    boundary_handlers: Dict[str, BoundaryAwareReplay] = field(default_factory=dict)
    
    # Container reference
    container: Optional['Container'] = None
    
    def initialize(self, container: 'Container') -> None:
        """Load signal indices and prepare for streaming."""
        self.container = container
        
        # Create storage manager and load indices
        self.storage_manager = SignalStorageManager(
            base_path=self.signal_storage_path,
            workflow_id=self.workflow_id
        )
        self.storage_manager.load_all()
        
        # Prepare signals for replay
        self._prepare_replay_data()
        
        # Initialize boundary handlers for each filtered strategy
        if self.regime_filter:
            for strategy_id in (self.strategy_filter or self.storage_manager.signal_indices.keys()):
                self.boundary_handlers[strategy_id] = BoundaryAwareReplay(
                    target_regime=self.regime_filter
                )
        
        logger.info(f"SignalStreamer initialized with {len(self.bar_indices)} bars "
                   f"containing signals from {len(self.storage_manager.signal_indices)} strategies")
    
    def start(self) -> None:
        """Start streaming."""
        self.current_index = 0
        logger.info("Signal streaming started")
    
    def stop(self) -> None:
        """Stop streaming."""
        logger.info(f"Signal streaming stopped at bar index {self.current_index}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get component state."""
        return {
            'current_index': self.current_index,
            'total_bars': len(self.bar_indices),
            'strategies_loaded': len(self.storage_manager.signal_indices),
            'signals_pending': sum(len(signals) for idx in self.bar_indices[self.current_index:] 
                                 for signals in self.signals_by_bar.get(idx, [])),
            'regime_filter': self.regime_filter,
            'strategy_filter': self.strategy_filter
        }
    
    def _prepare_replay_data(self) -> None:
        """Prepare signals for efficient replay."""
        # Collect all signals from filtered strategies
        for strategy_id, signal_index in self.storage_manager.signal_indices.items():
            # Apply strategy filter
            if self.strategy_filter and strategy_id not in self.strategy_filter:
                continue
            
            # Add signals to replay structure
            for signal in signal_index.signals:
                bar_idx = signal['bar_idx']
                if bar_idx not in self.signals_by_bar:
                    self.signals_by_bar[bar_idx] = []
                
                # Include strategy parameters with each signal
                signal_with_params = signal.copy()
                signal_with_params['strategy_params'] = signal_index.parameters
                signal_with_params['strategy_name'] = signal_index.strategy_name
                self.signals_by_bar[bar_idx].append(signal_with_params)
        
        # Sort bar indices for sequential replay
        self.bar_indices = sorted(self.signals_by_bar.keys())
        
        # If sparse replay, we only need bars with signals
        # If not sparse, we'd need to fill in all bars (not implemented here)
        if not self.sparse_replay:
            logger.warning("Non-sparse replay not implemented - using sparse mode")
    
    def has_more_bars(self) -> bool:
        """Check if more bars are available for replay."""
        return self.current_index < len(self.bar_indices)
    
    def get_next_bar_time(self) -> Optional[datetime]:
        """Get timestamp of next bar (if available in signal data)."""
        if not self.has_more_bars():
            return None
        
        bar_idx = self.bar_indices[self.current_index]
        signals = self.signals_by_bar.get(bar_idx, [])
        
        # Try to get timestamp from first signal
        if signals and 'timestamp' in signals[0]:
            return datetime.fromisoformat(signals[0]['timestamp'])
        
        return None
    
    def stream_next(self) -> bool:
        """
        Stream signals for next bar index.
        
        Returns:
            True if signals were streamed, False if no more bars
        """
        if not self.has_more_bars():
            return False
        
        bar_idx = self.bar_indices[self.current_index]
        signals = self.signals_by_bar.get(bar_idx, [])
        
        # Get current classifier states for this bar
        classifier_states = self._get_classifier_states_at_bar(bar_idx)
        
        # Process each signal
        emitted_count = 0
        for signal_data in signals:
            if self._should_emit_signal(signal_data, classifier_states):
                event = self._create_signal_event(signal_data, bar_idx, classifier_states)
                self.container.publish_event(event, target_scope='parent')
                emitted_count += 1
        
        logger.debug(f"Bar {bar_idx}: Emitted {emitted_count}/{len(signals)} signals")
        
        self.current_index += 1
        return True
    
    def _get_classifier_states_at_bar(self, bar_idx: int) -> Dict[str, str]:
        """Reconstruct classifier states at a specific bar."""
        states = {}
        
        for clf_name, clf_index in self.storage_manager.classifier_indices.items():
            state = clf_index.get_state_at_bar(bar_idx)
            if state:
                states[clf_name] = state
        
        return states
    
    def _should_emit_signal(self, signal_data: Dict[str, Any], 
                           classifier_states: Dict[str, str]) -> bool:
        """Check if signal should be emitted based on filters."""
        strategy_id = signal_data.get('strategy_id')
        
        # Check regime filter with boundary awareness
        if self.regime_filter and strategy_id in self.boundary_handlers:
            handler = self.boundary_handlers[strategy_id]
            current_regime = classifier_states.get('market_regime')  # Or specified classifier
            
            return handler.should_emit_signal(
                signal_data['bar_idx'],
                signal_data['value'],
                current_regime
            )
        
        return True
    
    def _create_signal_event(self, signal_data: Dict[str, Any], bar_idx: int,
                            classifier_states: Dict[str, str]) -> Event:
        """Create signal event from stored data."""
        # Determine direction from signal value
        direction = 'BUY' if signal_data['value'] > 0 else 'SELL'
        if signal_data['value'] == 0:
            direction = 'CLOSE'  # Exit signal
        
        # Build payload
        payload = {
            'symbol': signal_data['symbol'],
            'direction': direction,
            'strength': abs(signal_data['value']),
            'strategy_id': signal_data['strategy_id'],
            'strategy_name': signal_data.get('strategy_name', 'unknown'),
            'bar_idx': bar_idx,
            'classifier_states': signal_data.get('classifiers', classifier_states),
            'strategy_params': signal_data.get('strategy_params', {})
        }
        
        # Include bar data if available
        if 'bar_data' in signal_data:
            payload['bar_data'] = {signal_data['symbol']: signal_data['bar_data']}
        
        # Create event
        return Event(
            event_type=EventType.SIGNAL,
            payload=payload,
            source_id=self.container.container_id if self.container else 'signal_replay',
            metadata={
                'bar_idx': bar_idx,
                'replay_mode': 'sparse' if self.sparse_replay else 'full',
                'regime_filter': self.regime_filter,
                'timestamp': signal_data.get('timestamp', datetime.now().isoformat())
            }
        )
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of loaded signals."""
        summary = {
            'total_signals': sum(len(idx.signals) for idx in self.storage_manager.signal_indices.values()),
            'strategies': {},
            'classifiers': list(self.storage_manager.classifier_indices.keys()),
            'bar_range': {
                'first': self.bar_indices[0] if self.bar_indices else None,
                'last': self.bar_indices[-1] if self.bar_indices else None,
                'count': len(self.bar_indices)
            }
        }
        
        # Add per-strategy summary
        for strategy_id, index in self.storage_manager.signal_indices.items():
            summary['strategies'][strategy_id] = {
                'name': index.strategy_name,
                'signal_count': len(index.signals),
                'parameters': index.parameters
            }
        
        return summary