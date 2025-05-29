"""
Base strategy implementation with built-in optimization support.

All strategies in ADMF-PC are designed to be optimizable by default,
following the principle that optimization methods are built directly
into the base classes.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

from ...core.events import Event, EventType
from ..protocols import Strategy, Indicator, Feature, TradingRule


logger = logging.getLogger(__name__)


class StrategyBase:
    """
    Base class for trading strategies with built-in optimization support.
    
    This class provides:
    - Event handling for market data
    - Component composition (indicators, features, rules)
    - Built-in optimization interface
    - Signal generation and emission
    """
    
    def __init__(self, name: str = "strategy"):
        self.name = name
        
        # Components
        self.indicators: Dict[str, Indicator] = {}
        self.features: Dict[str, Feature] = {}
        self.rules: List[TradingRule] = []
        
        # State
        self.last_signal: Optional[str] = None
        self.signal_history: List[Dict[str, Any]] = []
        
        # Capabilities (added by capability system)
        self._lifecycle = None
        self._events = None
        self._optimization = None
    
    def add_indicator(self, name: str, indicator: Indicator) -> None:
        """Add an indicator component."""
        self.indicators[name] = indicator
    
    def add_feature(self, name: str, feature: Feature) -> None:
        """Add a feature component."""
        self.features[name] = feature
    
    def add_rule(self, rule: TradingRule) -> None:
        """Add a trading rule."""
        self.rules.append(rule)
    
    def setup_subscriptions(self) -> None:
        """Set up event subscriptions (called by event capability)."""
        if self._events:
            self._events.subscribe(EventType.BAR, self.on_bar)
    
    def on_bar(self, event: Event) -> None:
        """Handle new market bar."""
        bar_data = event.payload
        
        # Update indicators
        indicator_values = self._update_indicators(bar_data)
        
        # Calculate features
        feature_values = self._calculate_features(indicator_values)
        
        # Prepare data for rules
        rule_data = {
            **bar_data,
            'indicators': indicator_values,
            'features': feature_values
        }
        
        # Evaluate rules and generate signal
        signal = self._evaluate_rules(rule_data)
        
        if signal:
            self._emit_signal(signal)
    
    def _update_indicators(self, bar_data: Dict[str, Any]) -> Dict[str, float]:
        """Update all indicators and return their values."""
        values = {}
        
        price = bar_data.get('close', bar_data.get('price'))
        timestamp = bar_data.get('timestamp', datetime.now())
        
        for name, indicator in self.indicators.items():
            value = indicator.calculate(price, timestamp)
            if value is not None:
                values[name] = value
        
        return values
    
    def _calculate_features(self, indicator_values: Dict[str, float]) -> Dict[str, float]:
        """Calculate all features from indicator values."""
        values = {}
        
        for name, feature in self.features.items():
            if feature.ready:
                value = feature.calculate(indicator_values)
                if value is not None:
                    values[name] = value
        
        return values
    
    def _evaluate_rules(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate trading rules and generate signal."""
        # This is a template method - subclasses should override
        # Default implementation: majority vote
        
        if not self.rules:
            return None
        
        buy_strength = 0.0
        sell_strength = 0.0
        total_weight = 0.0
        
        for rule in self.rules:
            triggered, strength = rule.evaluate(data)
            if triggered:
                weight = rule.weight
                if strength > 0:
                    buy_strength += strength * weight
                else:
                    sell_strength += abs(strength) * weight
                total_weight += weight
        
        if total_weight == 0:
            return None
        
        # Normalize
        buy_strength /= total_weight
        sell_strength /= total_weight
        
        # Generate signal if strong enough
        threshold = 0.6  # Configurable
        
        if buy_strength > threshold and self.last_signal != "BUY":
            self.last_signal = "BUY"
            return self._create_signal("BUY", buy_strength, data)
        elif sell_strength > threshold and self.last_signal != "SELL":
            self.last_signal = "SELL"
            return self._create_signal("SELL", sell_strength, data)
        
        return None
    
    def _create_signal(self, direction: str, strength: float, 
                      data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a signal dictionary."""
        signal = {
            'symbol': data.get('symbol', 'UNKNOWN'),
            'direction': direction,
            'strength': strength,
            'price': data.get('close', data.get('price')),
            'timestamp': data.get('timestamp', datetime.now()),
            'strategy': self.name,
            'indicators': data.get('indicators', {}),
            'features': data.get('features', {}),
            'metadata': {
                'bar_data': data
            }
        }
        
        self.signal_history.append(signal)
        return signal
    
    def _emit_signal(self, signal: Dict[str, Any]) -> None:
        """Emit trading signal via event bus."""
        if hasattr(self, '_events') and self._events and self._events.event_bus:
            event = Event(EventType.SIGNAL, signal)
            self._events.event_bus.publish(event)
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.last_signal = None
        self.signal_history.clear()
        
        # Reset all components
        for indicator in self.indicators.values():
            if hasattr(indicator, 'reset'):
                indicator.reset()
        
        for feature in self.features.values():
            if hasattr(feature, 'reset'):
                feature.reset()
    
    # Built-in optimization interface
    def get_parameter_space(self) -> Dict[str, Any]:
        """
        Get parameter space for optimization.
        Default: empty (no parameters).
        Subclasses should override if they have parameters.
        """
        return {}
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Apply optimization parameters.
        Default: no-op.
        Subclasses should override if they have parameters.
        """
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current parameter values.
        Default: empty dict.
        Subclasses should override if they have parameters.
        """
        return {}
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate parameter values.
        Default: always valid.
        Subclasses should override for custom validation.
        """
        return True, ""