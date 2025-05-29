"""
Regime-aware strategy containers.

These containers encapsulate strategies that subscribe to a specific
regime classifier, enabling regime-adaptive trading.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

from ...core.events import Event
from ..protocols import Strategy, RegimeClassifier


logger = logging.getLogger(__name__)


class RegimeStrategyContainer:
    """
    Container that manages strategies based on regime classification.
    
    This container:
    - Subscribes to a specific regime classifier
    - Activates/deactivates strategies based on regime
    - Manages regime-specific parameter sets
    - Provides regime-aware signal generation
    """
    
    def __init__(self, name: str, classifier: RegimeClassifier):
        self.name = name
        self.classifier = classifier
        
        # Regime-specific strategies
        self.regime_strategies: Dict[str, List[Strategy]] = {}
        
        # Regime-specific parameters
        self.regime_parameters: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # Current state
        self.current_regime: Optional[str] = None
        self.active_strategies: List[Strategy] = []
        
        # Configuration
        self.enable_smooth_transition = True
        self.transition_period = 5  # bars
        self.transition_progress = 0
        
        # Capabilities
        self._lifecycle = None
        self._events = None
    
    def add_strategy_for_regime(self, regime: str, strategy: Strategy) -> None:
        """Add a strategy for a specific regime."""
        if regime not in self.regime_strategies:
            self.regime_strategies[regime] = []
        
        self.regime_strategies[regime].append(strategy)
        logger.info(f"Added strategy {strategy.name} for regime {regime}")
    
    def add_universal_strategy(self, strategy: Strategy) -> None:
        """Add a strategy that works in all regimes."""
        for regime in ['TRENDING_UP', 'TRENDING_DOWN', 'RANGE_BOUND', 'HIGH_VOLATILITY']:
            self.add_strategy_for_regime(regime, strategy)
    
    def set_regime_parameters(self, regime: str, strategy_name: str, 
                            parameters: Dict[str, Any]) -> None:
        """Set regime-specific parameters for a strategy."""
        if regime not in self.regime_parameters:
            self.regime_parameters[regime] = {}
        
        self.regime_parameters[regime][strategy_name] = parameters
    
    def setup_subscriptions(self) -> None:
        """Set up event subscriptions."""
        if self._events:
            # Subscribe to regime changes
            self._events.subscribe('REGIME_CHANGE', self.on_regime_change)
            
            # Subscribe to bars to forward to active strategies
            self._events.subscribe('BAR', self.on_bar)
            
            # Subscribe to indicator updates
            self._events.subscribe('INDICATOR_UPDATE', self.on_indicator_update)
    
    def on_regime_change(self, event: Event) -> None:
        """Handle regime change event."""
        data = event.payload
        
        # Only respond to our classifier
        if data.get('classifier') != type(self.classifier).__name__:
            return
        
        new_regime = data.get('new_regime')
        old_regime = data.get('old_regime')
        
        logger.info(f"Container {self.name} handling regime change: {old_regime} -> {new_regime}")
        
        # Handle regime transition
        self._transition_to_regime(new_regime)
    
    def on_bar(self, event: Event) -> None:
        """Forward bar events to active strategies."""
        # During transition, might need special handling
        if self.enable_smooth_transition and self.transition_progress > 0:
            self._handle_transition_bar(event)
        else:
            # Normal operation - forward to all active strategies
            for strategy in self.active_strategies:
                strategy.on_bar(event)
    
    def on_indicator_update(self, event: Event) -> None:
        """Forward indicator updates to strategies that need them."""
        # Strategies that need direct indicator access can subscribe
        # This is mainly for regime classifiers and advanced strategies
        pass
    
    def _transition_to_regime(self, new_regime: str) -> None:
        """Transition to a new regime."""
        self.current_regime = new_regime
        
        # Get strategies for new regime
        new_strategies = self.regime_strategies.get(new_regime, [])
        
        if self.enable_smooth_transition:
            # Start transition period
            self.transition_progress = self.transition_period
            self._start_transition(new_strategies)
        else:
            # Immediate switch
            self._switch_strategies(new_strategies)
    
    def _switch_strategies(self, new_strategies: List[Strategy]) -> None:
        """Immediately switch to new strategies."""
        # Deactivate old strategies
        for strategy in self.active_strategies:
            logger.info(f"Deactivating strategy: {strategy.name}")
        
        # Activate new strategies
        self.active_strategies = new_strategies.copy()
        
        # Apply regime-specific parameters
        if self.current_regime in self.regime_parameters:
            regime_params = self.regime_parameters[self.current_regime]
            
            for strategy in self.active_strategies:
                if strategy.name in regime_params:
                    params = regime_params[strategy.name]
                    strategy.set_parameters(params)
                    logger.info(f"Applied regime parameters to {strategy.name}: {params}")
        
        logger.info(f"Activated {len(self.active_strategies)} strategies for regime {self.current_regime}")
    
    def _start_transition(self, new_strategies: List[Strategy]) -> None:
        """Start smooth transition to new strategies."""
        # For now, immediate switch
        # Could implement gradual weight changes here
        self._switch_strategies(new_strategies)
    
    def _handle_transition_bar(self, event: Event) -> None:
        """Handle bar during transition period."""
        # Decrease transition progress
        self.transition_progress -= 1
        
        # Forward to active strategies
        for strategy in self.active_strategies:
            strategy.on_bar(event)
    
    def get_active_regime(self) -> Optional[str]:
        """Get currently active regime."""
        return self.current_regime
    
    def get_active_strategies(self) -> List[Strategy]:
        """Get currently active strategies."""
        return self.active_strategies.copy()
    
    def reset(self) -> None:
        """Reset container state."""
        self.current_regime = None
        self.active_strategies.clear()
        self.transition_progress = 0
        
        # Reset classifier
        if hasattr(self.classifier, 'reset'):
            self.classifier.reset()
        
        # Reset all strategies
        for strategies in self.regime_strategies.values():
            for strategy in strategies:
                if hasattr(strategy, 'reset'):
                    strategy.reset()


class AdaptiveEnsembleContainer:
    """
    Advanced container that manages an ensemble of strategies with
    regime-adaptive weights.
    """
    
    def __init__(self, name: str, classifier: RegimeClassifier):
        self.name = name
        self.classifier = classifier
        
        # Strategies with regime-specific weights
        self.strategies: List[Strategy] = []
        self.regime_weights: Dict[str, Dict[str, float]] = {}
        
        # Current state
        self.current_regime: Optional[str] = None
        self.current_weights: Dict[str, float] = {}
        
        # Signal aggregation
        self.min_agreement = 0.5  # Minimum agreement for signal
        self.signal_threshold = 0.6  # Minimum strength
        
        # Capabilities
        self._lifecycle = None
        self._events = None
    
    def add_strategy(self, strategy: Strategy, 
                    regime_weights: Optional[Dict[str, float]] = None) -> None:
        """
        Add strategy with optional regime-specific weights.
        
        Args:
            strategy: Strategy to add
            regime_weights: Dict mapping regime names to weights
        """
        self.strategies.append(strategy)
        
        if regime_weights:
            for regime, weight in regime_weights.items():
                if regime not in self.regime_weights:
                    self.regime_weights[regime] = {}
                self.regime_weights[regime][strategy.name] = weight
    
    def setup_subscriptions(self) -> None:
        """Set up event subscriptions."""
        if self._events:
            self._events.subscribe('REGIME_CHANGE', self.on_regime_change)
            self._events.subscribe('SIGNAL', self.on_signal)
    
    def on_regime_change(self, event: Event) -> None:
        """Handle regime change by updating weights."""
        data = event.payload
        
        if data.get('classifier') != type(self.classifier).__name__:
            return
        
        new_regime = data.get('new_regime')
        self.current_regime = new_regime
        
        # Update weights for new regime
        self._update_weights(new_regime)
    
    def on_signal(self, event: Event) -> None:
        """Collect and aggregate signals from strategies."""
        signal = event.payload
        
        # Check if signal is from one of our strategies
        strategy_name = signal.get('strategy')
        if not any(s.name == strategy_name for s in self.strategies):
            return
        
        # In a real implementation, would collect signals over a time window
        # and aggregate them. For now, just forward with regime-adjusted weight
        if strategy_name in self.current_weights:
            weight = self.current_weights[strategy_name]
            signal['ensemble_weight'] = weight
            signal['ensemble'] = self.name
            
            # Re-emit as ensemble signal
            if self._events and self._events.event_bus:
                ensemble_event = Event('ENSEMBLE_SIGNAL', signal)
                self._events.event_bus.publish(ensemble_event)
    
    def _update_weights(self, regime: str) -> None:
        """Update strategy weights for new regime."""
        self.current_weights.clear()
        
        if regime in self.regime_weights:
            self.current_weights = self.regime_weights[regime].copy()
        else:
            # Default equal weights
            equal_weight = 1.0 / len(self.strategies)
            for strategy in self.strategies:
                self.current_weights[strategy.name] = equal_weight
        
        logger.info(f"Updated ensemble weights for regime {regime}: {self.current_weights}")