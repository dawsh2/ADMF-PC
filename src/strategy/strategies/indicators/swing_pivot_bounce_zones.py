"""Swing Pivot Bounce with Zone-Based Stateless Logic."""
from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='swing_pivot_bounce_zones',
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
        'entry_zone': {'type': 'float', 'range': (0.001, 0.01), 'default': 0.003},  # Zone around S/R for entry
        'exit_zone': {'type': 'float', 'range': (0.001, 0.01), 'default': 0.003},   # Zone before opposite S/R for exit
        'min_range': {'type': 'float', 'range': (0.002, 0.02), 'default': 0.005}    # Minimum S/R range to trade
    },
    tags=['mean_reversion', 'support_resistance', 'bounce', 'zones', 'stateless']
)
def swing_pivot_bounce_zones(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Zone-based stateless support/resistance bounce strategy.
    
    Uses price zones around S/R levels to generate signals:
    - Entry zones: Within entry_zone % of S/R levels
    - Exit zones: Within exit_zone % of opposite S/R level
    - No position tracking needed - purely zone-based
    
    Signal logic:
    - 1: In support entry zone (not in resistance exit zone)
    - -1: In resistance entry zone (not in support exit zone) 
    - 0: In exit zones or neutral territory
    """
    sr_period = params.get('sr_period', 20)
    min_touches = params.get('min_touches', 2)
    entry_zone = params.get('entry_zone', 0.003)  # 0.3% default entry zone
    exit_zone = params.get('exit_zone', 0.003)    # 0.3% default exit zone
    min_range = params.get('min_range', 0.005)    # 0.5% minimum range
    
    # Get support/resistance features
    resistance = features.get(f'support_resistance_{sr_period}_{min_touches}_resistance')
    support = features.get(f'support_resistance_{sr_period}_{min_touches}_support')
    
    price = bar.get('close', 0)
    
    if resistance is None or support is None:
        return None
    
    # Check if range is wide enough to trade
    if resistance and support:
        range_pct = (resistance - support) / support
        if range_pct < min_range:
            # Range too narrow, don't trade
            signal_value = 0
        else:
            # Calculate zone memberships
            # Entry zones - where we look for bounces
            in_support_entry_zone = abs(price - support) / support <= entry_zone
            in_resistance_entry_zone = abs(price - resistance) / resistance <= entry_zone
            
            # Exit zones - where we take profits (approaching opposite level)
            # For longs: exit as we approach resistance from below
            in_resistance_exit_zone = (resistance - price) / price <= exit_zone and price < resistance
            
            # For shorts: exit as we approach support from above  
            in_support_exit_zone = (price - support) / price <= exit_zone and price > support
            
            # Calculate position in range for sustained signals
            position_in_range = (price - support) / (resistance - support)
            
            # Signal generation with sustained positions
            if in_resistance_exit_zone or in_support_exit_zone:
                # Exit zones take priority - flatten position
                signal_value = 0
            elif in_support_entry_zone or position_in_range < 0.4:
                # At support or in lower part of range - stay long
                signal_value = 1
            elif in_resistance_entry_zone or position_in_range > 0.6:
                # At resistance or in upper part of range - stay short
                signal_value = -1
            else:
                # Middle of range (40-60%) - no position
                signal_value = 0
    else:
        # Missing S/R data
        signal_value = 0
    
    # Always return current signal state
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    # Calculate zone boundaries for metadata
    support_entry_high = support * (1 + entry_zone) if support else None
    support_entry_low = support * (1 - entry_zone) if support else None
    resistance_entry_high = resistance * (1 + entry_zone) if resistance else None
    resistance_entry_low = resistance * (1 - entry_zone) if resistance else None
    
    resistance_exit_threshold = resistance * (1 - exit_zone) if resistance else None
    support_exit_threshold = support * (1 + exit_zone) if support else None
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'swing_pivot_bounce_zones',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'sr_period': sr_period,
            'min_touches': min_touches,
            'entry_zone': entry_zone,
            'exit_zone': exit_zone,
            'min_range': min_range,
            'price': price,
            'resistance': resistance if resistance else price * 1.01,
            'support': support if support else price * 0.99,
            'range_pct': (resistance - support) / support if resistance and support else 0,
            'in_support_entry': in_support_entry_zone if 'in_support_entry_zone' in locals() else False,
            'in_resistance_entry': in_resistance_entry_zone if 'in_resistance_entry_zone' in locals() else False,
            'in_resistance_exit': in_resistance_exit_zone if 'in_resistance_exit_zone' in locals() else False,
            'in_support_exit': in_support_exit_zone if 'in_support_exit_zone' in locals() else False,
            'support_entry_zone': [support_entry_low, support_entry_high],
            'resistance_entry_zone': [resistance_entry_low, resistance_entry_high],
            'resistance_exit_level': resistance_exit_threshold,
            'support_exit_level': support_exit_threshold
        }
    }