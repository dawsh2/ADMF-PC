"""Swing Pivot Bounce with Target-Based Exits."""
from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='swing_pivot_bounce_target',
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
        'target_threshold': {'type': 'float', 'range': (0.0005, 0.005), 'default': 0.002}  # Exit this % from target
    },
    tags=['mean_reversion', 'support_resistance', 'bounce', 'swing', 'levels', 'target_exit']
)
def swing_pivot_bounce_target(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Dynamic support/resistance bounce strategy with target-based exits.
    
    Entry logic:
    - Long when price bounces from support
    - Short when price bounces from resistance
    
    Exit logic:
    - Long exits when price reaches (resistance - target_threshold)
    - Short exits when price reaches (support + target_threshold)
    
    This allows capturing most of the move while exiting before hitting the opposite level.
    
    Returns sustained signal:
    - 1: Long position (bounced from support, targeting resistance)
    - -1: Short position (bounced from resistance, targeting support)
    - 0: Exit or no signal
    """
    sr_period = params.get('sr_period', 20)
    min_touches = params.get('min_touches', 2)
    bounce_threshold = params.get('bounce_threshold', 0.002)  # 0.2% proximity for bounce
    target_threshold = params.get('target_threshold', 0.002)  # Exit this % before target level
    
    # Get support/resistance features
    resistance = features.get(f'support_resistance_{sr_period}_{min_touches}_resistance')
    support = features.get(f'support_resistance_{sr_period}_{min_touches}_support')
    
    price = bar.get('close', 0)
    high = bar.get('high', price)
    low = bar.get('low', price)
    
    if resistance is None and support is None:
        return None
    
    # Initialize signal
    signal_value = 0
    
    # State tracking - in production this would be managed by the framework
    # For now, we'll determine state based on price position
    in_long = False
    in_short = False
    
    # Simple state detection based on price position
    if support and resistance:
        range_size = resistance - support
        position_in_range = (price - support) / range_size
        
        # If price is in lower 30% of range, likely in long from support bounce
        if position_in_range < 0.3:
            in_long = True
        # If price is in upper 30% of range, likely in short from resistance bounce
        elif position_in_range > 0.7:
            in_short = True
    
    # Exit logic - check if we're close to target
    if in_long and resistance:
        # Long position - exit when approaching resistance
        # Use target_threshold as percentage of price for consistent behavior
        long_target = resistance * (1 - target_threshold)
        if price >= long_target:
            signal_value = 0  # Exit long
        else:
            signal_value = 1  # Maintain long
            
    elif in_short and support:
        # Short position - exit when approaching support
        # Use target_threshold as percentage of price for consistent behavior
        short_target = support * (1 + target_threshold)
        if price <= short_target:
            signal_value = 0  # Exit short
        else:
            signal_value = -1  # Maintain short
    
    # Entry logic - only if not in position
    else:
        # Check for support bounce (long entry)
        if support and (low <= support * (1 + bounce_threshold) or 
                       abs(price - support) / support < bounce_threshold):
            # Additional check: make sure we have room to target
            if resistance and (resistance - price) / price > target_threshold * 2:
                signal_value = 1  # Long signal
                
        # Check for resistance bounce (short entry)
        elif resistance and (high >= resistance * (1 - bounce_threshold) or
                           abs(price - resistance) / resistance < bounce_threshold):
            # Additional check: make sure we have room to target
            if support and (price - support) / price > target_threshold * 2:
                signal_value = -1  # Short signal
    
    # Always return current signal state
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    # Calculate targets for metadata
    long_target = resistance * (1 - target_threshold) if resistance else None
    short_target = support * (1 + target_threshold) if support else None
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'swing_pivot_bounce_target',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'sr_period': sr_period,
            'min_touches': min_touches,
            'bounce_threshold': bounce_threshold,
            'target_threshold': target_threshold,
            'price': price,
            'resistance': resistance if resistance else price * 1.01,
            'support': support if support else price * 0.99,
            'long_target': long_target,
            'short_target': short_target,
            'support_distance': abs(price - support) / support if support else 0,
            'resistance_distance': abs(price - resistance) / resistance if resistance else 0,
            'to_long_target': abs(price - long_target) / price if long_target and signal_value == 1 else 0,
            'to_short_target': abs(price - short_target) / price if short_target and signal_value == -1 else 0
        }
    }