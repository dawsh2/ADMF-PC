"""Swing Pivot Bounce with Flexible Exit Positions."""
from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='swing_pivot_bounce_flex',
    feature_discovery=lambda params: [
        FeatureSpec('support_resistance', {
            'lookback': params.get('sr_period', 20),
            'min_touches': params.get('min_touches', 2)
        }, 'resistance'),
        FeatureSpec('support_resistance', {
            'lookback': params.get('sr_period', 20),
            'min_touches': params.get('min_touches', 2)
        }, 'support')
    ],
    parameter_space={
        'sr_period': {'type': 'int', 'range': (10, 100), 'default': 20},
        'min_touches': {'type': 'int', 'range': (2, 10), 'default': 2},
        'bounce_threshold': {'type': 'float', 'range': (0.001, 0.02), 'default': 0.002},
        'exit_threshold': {'type': 'float', 'range': (0.0005, 0.01), 'default': 0.001},
        'exit_position': {'type': 'float', 'range': (0.2, 0.8), 'default': 0.5}  # 0.5 = midpoint
    },
    tags=['mean_reversion', 'support_resistance', 'bounce', 'swing', 'levels', 'flexible_exit']
)
def swing_pivot_bounce_flex(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Dynamic support/resistance bounce strategy with flexible exit positions.
    
    Exit position parameter:
    - 0.2 = Exit 20% of the way from entry toward target (early exit)
    - 0.5 = Exit at midpoint between S/R (default)
    - 0.8 = Exit 80% of the way to target (late exit)
    
    Returns sustained signal based on level bounces:
    - 1: Price bounces from support level (expect move up)
    - -1: Price bounces from resistance level (expect move down)
    - 0: Exit or no signal
    """
    sr_period = params.get('sr_period', 20)
    min_touches = params.get('min_touches', 2)
    bounce_threshold = params.get('bounce_threshold', 0.002)  # 0.2% proximity for bounce
    exit_threshold = params.get('exit_threshold', 0.001)  # 0.1% tolerance for exit level
    exit_position = params.get('exit_position', 0.5)  # Where to exit between S/R
    
    # Get support/resistance features
    resistance = features.get(f'support_resistance_{sr_period}_{min_touches}_resistance')
    support = features.get(f'support_resistance_{sr_period}_{min_touches}_support')
    
    price = bar.get('close', 0)
    high = bar.get('high', price)
    low = bar.get('low', price)
    
    if resistance is None and support is None:
        return None
    
    # Calculate exit level based on exit_position
    exit_level = None
    if support and resistance:
        # Exit level is exit_position fraction between support and resistance
        exit_level = support + (resistance - support) * exit_position
    
    # Get previous signal to maintain state
    signal_value = 0
    
    # First, check if we should exit at our exit level
    if exit_level:
        if abs(price - exit_level) / exit_level <= exit_threshold:
            # Price is within exit threshold of exit level - exit any position
            signal_value = 0
        else:
            # Determine position based on which side of exit level we're on
            if price < exit_level:
                # Below exit level - check for long entry at support
                if support and (low <= support * (1 + bounce_threshold) or 
                               abs(price - support) / support < bounce_threshold / 2):
                    signal_value = 1  # Long signal
                else:
                    signal_value = 0  # No signal
            else:
                # Above exit level - check for short entry at resistance
                if resistance and (high >= resistance * (1 - bounce_threshold) or
                                 abs(price - resistance) / resistance < bounce_threshold / 2):
                    signal_value = -1  # Short signal
                else:
                    signal_value = 0  # No signal
    else:
        # No exit level available (only one level) - use original logic
        if support and (low <= support * (1 + bounce_threshold) or
                       abs(price - support) / support < bounce_threshold / 2):
            signal_value = 1
        elif resistance and (high >= resistance * (1 - bounce_threshold) or
                           abs(price - resistance) / resistance < bounce_threshold / 2):
            signal_value = -1
    
    # Always return current signal state
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'swing_pivot_bounce_flex',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'sr_period': sr_period,
            'min_touches': min_touches,
            'bounce_threshold': bounce_threshold,
            'exit_threshold': exit_threshold,
            'exit_position': exit_position,
            'price': price,
            'resistance': resistance if resistance else price * 1.01,
            'support': support if support else price * 0.99,
            'exit_level': exit_level if exit_level else price,
            'support_distance': abs(price - support) / support if support else 0,
            'resistance_distance': abs(price - resistance) / resistance if resistance else 0,
            'exit_distance': abs(price - exit_level) / exit_level if exit_level else 0
        }
    }