"""
Classifier-aware strategy containers.

These containers encapsulate strategies that subscribe to a specific
classifier, enabling adaptive trading based on market classifications.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

from ...core.events import Event
from ..protocols import Strategy, Classifier


logger = logging.getLogger(__name__)


class ClassifierContainer:
    """
    Container that manages strategies based on market classification.
    
    This container:
    - Subscribes to a specific classifier
    - Activates/deactivates strategies based on classification
    - Manages classification-specific parameter sets
    - Provides classification-aware signal generation
    """
    
    def __init__(self, name: str, classifier: Classifier):
        self.name = name
        self.classifier = classifier
        
        # Class-specific strategies
        self.class_strategies: Dict[str, List[Strategy]] = {}
        
        # Class-specific parameters
        self.class_parameters: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # Current state
        self.current_class: Optional[str] = None
        self.active_strategies: List[Strategy] = []
        
        # Configuration
        self.enable_smooth_transition = True
        self.transition_period = 5  # bars
        self.transition_progress = 0
        
        # Capabilities
        self._lifecycle = None
        self._events = None
    
    def add_strategy_for_class(self, class_name: str, strategy: Strategy) -> None:
        """Add a strategy for a specific class."""
        if class_name not in self.class_strategies:
            self.class_strategies[class_name] = []
        
        self.class_strategies[class_name].append(strategy)
        logger.info(f"Added strategy {strategy.name} for class {class_name}")
    
    def add_universal_strategy(self, strategy: Strategy) -> None:
        """Add a strategy that works in all classifications."""
        for class_name in ['TRENDING_UP', 'TRENDING_DOWN', 'RANGE_BOUND', 'HIGH_VOLATILITY']:
            self.add_strategy_for_class(class_name, strategy)
    
    def set_class_parameters(self, class_name: str, strategy_name: str, 
                            parameters: Dict[str, Any]) -> None:
        """Set class-specific parameters for a strategy."""
        if class_name not in self.class_parameters:
            self.class_parameters[class_name] = {}
        
        self.class_parameters[class_name][strategy_name] = parameters
    
    def setup_subscriptions(self) -> None:
        """Set up event subscriptions."""
        if self._events:
            # Subscribe to classification changes
            self._events.subscribe('CLASSIFICATION_CHANGE', self.on_classification_change)
            
            # Subscribe to bars to forward to active strategies
            self._events.subscribe('BAR', self.on_bar)
            
            # Subscribe to indicator updates
            self._events.subscribe('INDICATOR_UPDATE', self.on_indicator_update)
    
    def on_classification_change(self, event: Event) -> None:
        """Handle classification change event."""
        data = event.payload
        
        # Only respond to our classifier
        if data.get('classifier') != type(self.classifier).__name__:
            return
        
        new_class = data.get('new_class')
        old_class = data.get('old_class')
        
        logger.info(f"Container {self.name} handling classification change: {old_class} -> {new_class}")
        
        # Handle class transition
        self._transition_to_class(new_class)
    
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
        # This is mainly for classifiers and advanced strategies
        pass
    
    def _transition_to_class(self, new_class: str) -> None:
        """Transition to a new classification."""
        self.current_class = new_class
        
        # Get strategies for new class
        new_strategies = self.class_strategies.get(new_class, [])
        
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
        
        # Apply class-specific parameters
        if self.current_class in self.class_parameters:
            class_params = self.class_parameters[self.current_class]
            
            for strategy in self.active_strategies:
                if strategy.name in class_params:
                    params = class_params[strategy.name]
                    strategy.set_parameters(params)
                    logger.info(f"Applied class parameters to {strategy.name}: {params}")
        
        logger.info(f"Activated {len(self.active_strategies)} strategies for class {self.current_class}")
    
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
    
    def get_active_class(self) -> Optional[str]:
        """Get currently active classification."""
        return self.current_class
    
    def get_active_strategies(self) -> List[Strategy]:
        """Get currently active strategies."""
        return self.active_strategies.copy()
    
    def reset(self) -> None:
        """Reset container state."""
        self.current_class = None
        self.active_strategies.clear()
        self.transition_progress = 0
        
        # Reset classifier
        if hasattr(self.classifier, 'reset'):
            self.classifier.reset()
        
        # Reset all strategies
        for strategies in self.class_strategies.values():
            for strategy in strategies:
                if hasattr(strategy, 'reset'):
                    strategy.reset()


class AdaptiveWeightContainer:
    """
    Container that manages strategies with classification-adaptive weights.
    
    This container allows strategies to have different weights based on
    the current market classification, enabling dynamic portfolio allocation.
    """
    
    def __init__(self, name: str, classifier: Classifier):
        self.name = name
        self.classifier = classifier
        
        # Strategies with class-specific weights
        self.strategies: List[Strategy] = []
        self.class_weights: Dict[str, Dict[str, float]] = {}
        
        # Current state
        self.current_class: Optional[str] = None
        self.current_weights: Dict[str, float] = {}
        
        # Signal aggregation
        self.min_agreement = 0.5  # Minimum agreement for signal
        self.signal_threshold = 0.6  # Minimum strength
        
        # Capabilities
        self._lifecycle = None
        self._events = None
    
    def add_strategy(self, strategy: Strategy, 
                    class_weights: Optional[Dict[str, float]] = None) -> None:
        """
        Add strategy with optional class-specific weights.
        
        Args:
            strategy: Strategy to add
            class_weights: Dict mapping class names to weights
        """
        self.strategies.append(strategy)
        
        if class_weights:
            for class_name, weight in class_weights.items():
                if class_name not in self.class_weights:
                    self.class_weights[class_name] = {}
                self.class_weights[class_name][strategy.name] = weight
    
    def setup_subscriptions(self) -> None:
        """Set up event subscriptions."""
        if self._events:
            self._events.subscribe('CLASSIFICATION_CHANGE', self.on_classification_change)
            self._events.subscribe('SIGNAL', self.on_signal)
    
    def on_classification_change(self, event: Event) -> None:
        """Handle classification change by updating weights."""
        data = event.payload
        
        if data.get('classifier') != type(self.classifier).__name__:
            return
        
        new_class = data.get('new_class')
        self.current_class = new_class
        
        # Update weights for new class
        self._update_weights(new_class)
    
    def on_signal(self, event: Event) -> None:
        """Collect and aggregate signals from strategies."""
        signal = event.payload
        
        # Check if signal is from one of our strategies
        strategy_name = signal.get('strategy')
        if not any(s.name == strategy_name for s in self.strategies):
            return
        
        # In a real implementation, would collect signals over a time window
        # and aggregate them. For now, just forward with class-adjusted weight
        if strategy_name in self.current_weights:
            weight = self.current_weights[strategy_name]
            signal['adaptive_weight'] = weight
            signal['container'] = self.name
            
            # Re-emit as weighted signal
            if self._events and self._events.event_bus:
                weighted_event = Event('WEIGHTED_SIGNAL', signal)
                self._events.event_bus.publish(weighted_event)
    
    def _update_weights(self, class_name: str) -> None:
        """Update strategy weights for new classification."""
        self.current_weights.clear()
        
        if class_name in self.class_weights:
            self.current_weights = self.class_weights[class_name].copy()
        else:
            # Default equal weights
            equal_weight = 1.0 / len(self.strategies)
            for strategy in self.strategies:
                self.current_weights[strategy.name] = equal_weight
        
        logger.info(f"Updated adaptive weights for class {class_name}: {self.current_weights}")