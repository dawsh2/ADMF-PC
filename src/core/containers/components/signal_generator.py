"""
Signal generation component for Feature Container.
Integrates TimeAlignmentBuffer with stateless functions and sparse storage.

This component is added to Feature containers to enable signal generation
with automatic storage of signals and classifier state changes.
"""

from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime

from ...events import Event, EventType
from ...storage.signals import SignalStorageManager, SignalIndex, ClassifierChangeIndex
from ..protocols import ContainerComponent


logger = logging.getLogger(__name__)


@dataclass
class SignalGeneratorComponent(ContainerComponent):
    """
    Generates signals using stateless classifier/strategy functions.
    
    This component:
    1. Manages stateless classifier and strategy functions
    2. Tracks classifier state changes for sparse storage
    3. Generates signal events with full context
    4. Optionally stores signals for later replay
    """
    
    # Configuration
    storage_enabled: bool = False
    storage_path: Optional[Path] = None
    workflow_id: Optional[str] = None
    
    # Stateless function registries
    classifiers: Dict[str, Callable] = field(default_factory=dict)
    strategies: Dict[str, Callable] = field(default_factory=dict)
    
    # Strategy metadata (parameters for each strategy_id)
    strategy_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Storage manager
    storage_manager: Optional[SignalStorageManager] = None
    
    # Current classifier states
    current_states: Dict[str, str] = field(default_factory=dict)
    
    # Container reference
    container: Optional['Container'] = None
    
    def initialize(self, container: 'Container') -> None:
        """Initialize component with container reference."""
        self.container = container
        
        # Setup storage if enabled
        if self.storage_enabled and self.storage_path:
            self.storage_manager = SignalStorageManager(
                base_path=Path(self.storage_path),
                workflow_id=self.workflow_id or 'default'
            )
            logger.info(f"Signal storage enabled at {self.storage_path}")
    
    def start(self) -> None:
        """Start the component."""
        logger.info(f"SignalGeneratorComponent started with {len(self.classifiers)} classifiers "
                   f"and {len(self.strategies)} strategies")
    
    def stop(self) -> None:
        """Stop component and save any pending data."""
        if self.storage_manager:
            self.storage_manager.save_all()
            logger.info("Saved all signal and classifier data")
    
    def get_state(self) -> Dict[str, Any]:
        """Get component state."""
        return {
            'classifiers': list(self.classifiers.keys()),
            'strategies': list(self.strategies.keys()),
            'current_classifier_states': self.current_states.copy(),
            'storage_enabled': self.storage_enabled,
            'signals_stored': sum(len(idx.signals) for idx in self.storage_manager.signal_indices.values()) 
                             if self.storage_manager else 0
        }
    
    def register_classifier(self, name: str, func: Callable) -> None:
        """Register a stateless classifier function."""
        self.classifiers[name] = func
        logger.debug(f"Registered classifier: {name}")
    
    def register_strategy(self, name: str, strategy_id: str, 
                         func: Callable, parameters: Dict[str, Any]) -> None:
        """Register a stateless strategy function with parameters."""
        self.strategies[strategy_id] = func
        self.strategy_configs[strategy_id] = {
            'name': name,
            'parameters': parameters,
            'function': func
        }
        logger.debug(f"Registered strategy: {name} with id {strategy_id}")
    
    def process_synchronized_bars(self, bars: Dict[str, Any], features: Dict[str, Any]) -> List[Event]:
        """
        Process synchronized bars through classifiers and strategies.
        
        Args:
            bars: Dict of symbol -> bar data
            features: Dict of symbol -> calculated features
            
        Returns:
            List of signal events to publish
        """
        events = []
        
        # Get primary bar for indexing (first symbol)
        primary_symbol = list(bars.keys())[0]
        primary_bar = bars[primary_symbol]
        bar_idx = primary_bar.get('index', 0)
        
        # 1. Run classifiers and track changes
        classifier_states = self._process_classifiers(features, bar_idx)
        
        # 2. Run strategies with classifier context
        signal_events = self._process_strategies(features, classifier_states, bars, bar_idx)
        events.extend(signal_events)
        
        return events
    
    def _process_classifiers(self, features: Dict[str, Any], bar_idx: int) -> Dict[str, str]:
        """Run classifiers and track state changes."""
        classifier_states = {}
        
        for clf_name, clf_func in self.classifiers.items():
            try:
                # Call stateless classifier
                new_state = clf_func(features)
                old_state = self.current_states.get(clf_name)
                
                # Track state change
                if new_state != old_state:
                    logger.debug(f"Classifier {clf_name} changed: {old_state} -> {new_state}")
                    
                    # Store change if storage enabled
                    if self.storage_manager:
                        clf_index = self.storage_manager.get_or_create_classifier_index(clf_name)
                        clf_index.record_change(bar_idx, old_state or 'NONE', new_state)
                    
                    self.current_states[clf_name] = new_state
                
                classifier_states[clf_name] = new_state
                
            except Exception as e:
                logger.error(f"Error in classifier {clf_name}: {e}")
                # Use previous state on error
                classifier_states[clf_name] = self.current_states.get(clf_name, 'UNKNOWN')
        
        return classifier_states
    
    def _process_strategies(self, features: Dict[str, Any], classifier_states: Dict[str, str],
                           bars: Dict[str, Any], bar_idx: int) -> List[Event]:
        """Run strategies and generate signal events."""
        events = []
        
        for strategy_id, strategy_config in self.strategy_configs.items():
            try:
                strategy_func = strategy_config['function']
                strategy_name = strategy_config['name']
                parameters = strategy_config['parameters']
                
                # Call stateless strategy
                signal = strategy_func(features, classifier_states, parameters)
                
                if signal and signal.get('value', 0) != 0:
                    # Create signal event
                    event = self._create_signal_event(
                        signal, strategy_id, strategy_name, 
                        bars, features, classifier_states, bar_idx
                    )
                    events.append(event)
                    
                    # Store signal if storage enabled
                    if self.storage_manager:
                        self._store_signal(
                            strategy_id, strategy_name, parameters,
                            signal, bars, classifier_states, bar_idx
                        )
                        
            except Exception as e:
                logger.error(f"Error in strategy {strategy_id}: {e}")
        
        return events
    
    def _create_signal_event(self, signal: Dict[str, Any], strategy_id: str,
                            strategy_name: str, bars: Dict[str, Any],
                            features: Dict[str, Any], classifier_states: Dict[str, str],
                            bar_idx: int) -> Event:
        """Create a signal event with full context."""
        # Extract signal details
        symbol = signal.get('symbol')
        direction = 'BUY' if signal.get('value', 0) > 0 else 'SELL'
        strength = abs(signal.get('value', 0))
        
        # Build payload with context
        payload = {
            'symbol': symbol,
            'direction': direction,
            'strength': strength,
            'strategy_id': strategy_id,
            'strategy_name': strategy_name,
            'bar_idx': bar_idx,
            'classifier_states': classifier_states.copy(),
            'bar_data': {}
        }
        
        # Add minimal bar data for each symbol
        for sym, bar in bars.items():
            payload['bar_data'][sym] = {
                'open': bar.get('open'),
                'high': bar.get('high'),
                'low': bar.get('low'),
                'close': bar.get('close'),
                'volume': bar.get('volume'),
                'timestamp': bar.get('timestamp')
            }
        
        # Add relevant features
        if symbol in features:
            payload['features'] = features[symbol]
        
        # Add any additional signal metadata
        if 'metadata' in signal:
            payload['metadata'] = signal['metadata']
        
        # Create event
        return Event(
            event_type=EventType.SIGNAL,
            payload=payload,
            source_id=self.container.container_id if self.container else 'signal_generator',
            metadata={
                'bar_idx': bar_idx,
                'strategy_id': strategy_id,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def _store_signal(self, strategy_id: str, strategy_name: str,
                     parameters: Dict[str, Any], signal: Dict[str, Any],
                     bars: Dict[str, Any], classifier_states: Dict[str, str],
                     bar_idx: int) -> None:
        """Store signal in sparse index."""
        # Get or create signal index
        signal_index = self.storage_manager.get_or_create_signal_index(
            strategy_id, strategy_name, parameters
        )
        
        # Get primary symbol/timeframe
        symbol = signal.get('symbol')
        bar_data = bars.get(symbol, {})
        timeframe = bar_data.get('timeframe', '1m')
        
        # Append signal
        signal_index.append_signal(
            bar_idx=bar_idx,
            signal_value=signal.get('value', 0),
            symbol=symbol,
            timeframe=timeframe,
            classifier_states=classifier_states,
            bar_data=bar_data
        )