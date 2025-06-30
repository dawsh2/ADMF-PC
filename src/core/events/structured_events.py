"""
Structured event system with embedded metadata.

This module provides enhanced event creation that embeds parameters and metadata
directly in events, eliminating the need for string parsing and enabling
flexible subscription patterns.
"""

from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from .types import Event, EventType


def create_structured_signal_event(
    symbol: str,
    timeframe: str,
    direction: str,
    strength: float,
    strategy_type: str,
    parameters: Dict[str, Any],
    source_id: Optional[str] = None,
    container_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Event:
    """
    Create a structured signal event with embedded parameters.
    
    Instead of encoding everything in strategy_id like:
        "SPY_1m_sma_crossover_grid_5_20"
    
    We have structured data:
        {
            'symbol': 'SPY',
            'timeframe': '1m',
            'strategy_type': 'sma_crossover',
            'parameters': {'fast_period': 5, 'slow_period': 20}
        }
    
    Args:
        symbol: Trading symbol
        timeframe: Time frame (1m, 5m, etc)
        direction: Signal direction (long/short/neutral)
        strength: Signal strength [0, 1]
        strategy_type: Base strategy type (e.g., 'sma_crossover')
        parameters: Strategy parameters as dict
        source_id: Source component ID
        container_id: Container ID
        metadata: Additional metadata
        
    Returns:
        Structured signal event
    """
    # Build structured payload
    payload = {
        'symbol': symbol,
        'timeframe': timeframe,
        'direction': direction,
        'strength': strength,
        'strategy_type': strategy_type,
        'parameters': parameters,
        
        # Legacy compatibility - can be removed later
        'strategy_id': _generate_legacy_id(symbol, timeframe, strategy_type, parameters),
        
        # Additional context
        'timestamp': datetime.now().isoformat(),
    }
    
    # Merge any additional metadata
    if metadata:
        payload['metadata'] = metadata
    
    return Event(
        event_type=EventType.SIGNAL.value,
        payload=payload,
        source_id=source_id,
        container_id=container_id,
        metadata={
            'category': 'trading',
            'version': '2.0'  # Structured event version
        }
    )


def _generate_legacy_id(symbol: str, timeframe: str, strategy_type: str, 
                       parameters: Dict[str, Any]) -> str:
    """
    Generate legacy strategy_id for backward compatibility.
    
    Can be removed once all consumers are updated to use structured data.
    """
    # Sort parameters for consistent naming
    param_parts = []
    for key, value in sorted(parameters.items()):
        param_parts.append(str(value))
    
    param_str = '_'.join(param_parts)
    return f"{symbol}_{timeframe}_{strategy_type}_{param_str}"


class SubscriptionDescriptor:
    """
    Descriptor for structured event subscriptions.
    
    Allows flexible matching patterns for event routing.
    """
    
    def __init__(self, criteria: Dict[str, Any]):
        """
        Initialize subscription descriptor.
        
        Args:
            criteria: Dict of criteria to match against event payload
            
        Example:
            # Subscribe to specific strategy and parameters
            SubscriptionDescriptor({
                'symbol': 'SPY',
                'strategy_type': 'sma_crossover',
                'parameters': {'fast_period': 5, 'slow_period': 20}
            })
            
            # Subscribe to all SMA crossovers on SPY
            SubscriptionDescriptor({
                'symbol': 'SPY',
                'strategy_type': 'sma_crossover'
            })
            
            # Subscribe to all signals with fast_period=5
            SubscriptionDescriptor({
                'parameters.fast_period': 5
            })
        """
        self.criteria = criteria
    
    def matches(self, event: Event) -> bool:
        """
        Check if event matches this subscription.
        
        Supports:
        - Exact matching
        - Nested field matching (dot notation)
        - List membership
        - Partial dict matching
        """
        if event.event_type != EventType.SIGNAL.value:
            return False
            
        payload = event.payload
        
        for key, expected_value in self.criteria.items():
            # Handle nested keys (e.g., 'parameters.fast_period')
            if '.' in key:
                actual_value = self._get_nested_value(payload, key)
            else:
                actual_value = payload.get(key)
            
            # Check matching
            if not self._values_match(actual_value, expected_value):
                return False
        
        return True
    
    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Any:
        """Get value from nested dict using dot notation."""
        parts = key.split('.')
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
    def _values_match(self, actual: Any, expected: Any) -> bool:
        """
        Check if values match according to subscription rules.
        
        Rules:
        - None matches nothing (explicit None check)
        - Lists: actual must be in expected list
        - Dicts: all expected keys must match in actual
        - Other: exact equality
        """
        if actual is None:
            return False
        
        # List membership
        if isinstance(expected, list):
            return actual in expected
        
        # Partial dict matching
        if isinstance(expected, dict) and isinstance(actual, dict):
            for k, v in expected.items():
                if k not in actual or not self._values_match(actual[k], v):
                    return False
            return True
        
        # Exact match
        return actual == expected
    
    def __repr__(self):
        return f"SubscriptionDescriptor({self.criteria})"


class StructuredEventFilter:
    """
    Filter for structured events in the event bus.
    
    Can be used with event bus subscriptions.
    """
    
    def __init__(self, descriptor: SubscriptionDescriptor):
        self.descriptor = descriptor
    
    def __call__(self, event: Event) -> bool:
        """Make filter callable for event bus."""
        return self.descriptor.matches(event)


def create_subscription_filter(**criteria) -> StructuredEventFilter:
    """
    Convenience function to create event filters.
    
    Example:
        # Filter for specific strategy
        filter = create_subscription_filter(
            symbol='SPY',
            strategy_type='sma_crossover',
            parameters={'fast_period': 5}
        )
        
        event_bus.subscribe(EventType.SIGNAL, handler, filter)
    """
    descriptor = SubscriptionDescriptor(criteria)
    return StructuredEventFilter(descriptor)


# Enhanced classification event with structured data
def create_structured_classification_event(
    symbol: str,
    timeframe: str,
    regime: str,
    confidence: float,
    classifier_type: str,
    parameters: Dict[str, Any],
    previous_regime: Optional[str] = None,
    features: Optional[Dict[str, float]] = None,
    source_id: Optional[str] = None,
    container_id: Optional[str] = None
) -> Event:
    """
    Create structured classification event.
    
    Similar improvements as signal events - embedded parameters instead of
    string IDs.
    """
    payload = {
        'symbol': symbol,
        'timeframe': timeframe,
        'regime': regime,
        'confidence': confidence,
        'classifier_type': classifier_type,
        'parameters': parameters,
        'previous_regime': previous_regime,
        'features': features or {},
        'is_regime_change': previous_regime is not None and previous_regime != regime,
        
        # Legacy compatibility
        'classifier_id': f"{symbol}_{timeframe}_{classifier_type}",
        
        'timestamp': datetime.now().isoformat(),
    }
    
    return Event(
        event_type=EventType.CLASSIFICATION.value,
        payload=payload,
        source_id=source_id,
        container_id=container_id,
        metadata={
            'category': 'classification',
            'version': '2.0'
        }
    )