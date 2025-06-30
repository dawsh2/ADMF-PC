"""
Market structure indicator strategies.

All structure strategies that generate signals based on support/resistance,
pivot points, and price patterns.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='pivot_points',
    feature_discovery=lambda params: [
        FeatureSpec('pivot_points', {}, 'pivot'),
        FeatureSpec('pivot_points', {}, 'r1'),
        FeatureSpec('pivot_points', {}, 's1')
    ],
    parameter_space={}
)
def pivot_points(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Pivot point support/resistance strategy.
    
    Returns sustained signal based on price vs pivot levels:
    - 1: Price breaks above R1 (bullish breakout)
    - -1: Price breaks below S1 (bearish breakdown)
    - 0: Price between S1 and R1 (ranging)
    """
    # Get features - pivot_points returns a dict with pivot, r1, s1, etc.
    pivot = features.get('pivot_points_pivot')
    r1 = features.get('pivot_points_r1')
    s1 = features.get('pivot_points_s1')
    price = bar.get('close', 0)
    
    if pivot is None or r1 is None or s1 is None:
        return None
    
    # Determine signal based on pivot levels
    if price > r1:
        signal_value = 1   # Bullish breakout
    elif price < s1:
        signal_value = -1  # Bearish breakdown
    else:
        signal_value = 0   # Ranging between S1 and R1
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'pivot_points',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'price': price,                           # Values for analysis
            'pivot': pivot,
            'r1': r1,
            's1': s1,
            'position': 'above_r1' if price > r1 else 'below_s1' if price < s1 else 'neutral'
        }
    }


@strategy(
    name='pivot_bounces',
    feature_discovery=lambda params: [
        FeatureSpec('pivot_points', {}, 'pivot'),
        FeatureSpec('pivot_points', {}, 'r1'),
        FeatureSpec('pivot_points', {}, 's1'),
        FeatureSpec('pivot_points', {}, 'r2'),
        FeatureSpec('pivot_points', {}, 's2')
    ],
    parameter_space={
        'touch_threshold': {'type': 'float', 'range': (0.0001, 0.001), 'default': 0.0005},  # 0.05% default
        'use_extended_levels': {'type': 'bool', 'default': True},  # Whether to use S2/R2 levels
        'left_lookback': {'type': 'int', 'range': (3, 20), 'default': 10},  # Bars to look left for pivot
        'right_lookback': {'type': 'int', 'range': (3, 20), 'default': 10},  # Bars to look right for pivot
        'channel_width': {'type': 'float', 'range': (0.001, 0.03), 'default': 0.01},  # 0.1% to 3% channel
        'use_channel': {'type': 'bool', 'default': False}  # Whether to use channel instead of exact levels
    },
    strategy_type='mean_reversion',
    tags=['mean_reversion', 'structure', 'pivot', 'support_resistance']
)
def pivot_bounces(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Pivot point bounce (mean reversion) strategy.
    
    Returns sustained signal expecting bounces at pivot levels:
    - 1: Price near S1 or S2 (expect bounce up from support)
    - -1: Price near R1 or R2 (expect bounce down from resistance)
    - 0: Price away from key levels
    
    This is the opposite of pivot_points which trades breakouts.
    """
    touch_threshold = params.get('touch_threshold', 0.0005)  # 0.05%
    use_extended_levels = params.get('use_extended_levels', True)
    left_lookback = params.get('left_lookback', 10)
    right_lookback = params.get('right_lookback', 10)
    channel_width = params.get('channel_width', 0.01)  # 1% default
    use_channel = params.get('use_channel', False)
    
    # Get features - standard pivot points
    pivot = features.get('pivot_points_pivot')
    r1 = features.get('pivot_points_r1')
    s1 = features.get('pivot_points_s1')
    r2 = features.get('pivot_points_r2') if use_extended_levels else None
    s2 = features.get('pivot_points_s2') if use_extended_levels else None
    price = bar.get('close', 0)
    
    if pivot is None or r1 is None or s1 is None:
        return None
    
    signal_value = 0
    
    if use_channel:
        # Channel mode: Create bands around pivot levels
        # Check if price is in support channel (expect bounce up)
        if s1 is not None:
            s1_upper = s1 * (1 + channel_width)
            s1_lower = s1 * (1 - channel_width)
            if s1_lower <= price <= s1_upper:
                signal_value = 1  # In S1 channel
        
        if signal_value == 0 and s2 is not None and use_extended_levels:
            s2_upper = s2 * (1 + channel_width)
            s2_lower = s2 * (1 - channel_width)
            if s2_lower <= price <= s2_upper:
                signal_value = 1  # In S2 channel
        
        # Check if price is in resistance channel (expect bounce down)
        if signal_value == 0 and r1 is not None:
            r1_upper = r1 * (1 + channel_width)
            r1_lower = r1 * (1 - channel_width)
            if r1_lower <= price <= r1_upper:
                signal_value = -1  # In R1 channel
        
        if signal_value == 0 and r2 is not None and use_extended_levels:
            r2_upper = r2 * (1 + channel_width)
            r2_lower = r2 * (1 - channel_width)
            if r2_lower <= price <= r2_upper:
                signal_value = -1  # In R2 channel
    else:
        # Touch mode: Original logic with exact threshold
        # Check if price is near support levels (expect bounce up)
        if s1 is not None and abs(price - s1) / s1 <= touch_threshold:
            signal_value = 1  # Buy at S1 support
        elif s2 is not None and use_extended_levels and abs(price - s2) / s2 <= touch_threshold:
            signal_value = 1  # Buy at S2 support
        
        # Check if price is near resistance levels (expect bounce down)
        elif r1 is not None and abs(price - r1) / r1 <= touch_threshold:
            signal_value = -1  # Sell at R1 resistance
        elif r2 is not None and use_extended_levels and abs(price - r2) / r2 <= touch_threshold:
            signal_value = -1  # Sell at R2 resistance
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'pivot_bounces',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'touch_threshold': touch_threshold,           # Parameters for sparse storage
            'left_lookback': left_lookback,
            'right_lookback': right_lookback,
            'channel_width': channel_width if use_channel else None,
            'use_channel': use_channel,
            'price': price,                              # Values for analysis
            'pivot': pivot,
            'r1': r1,
            's1': s1,
            'r2': r2,
            's2': s2,
            'nearest_level': _find_nearest_pivot_level(price, pivot, r1, s1, r2, s2),
            'distance_to_nearest': _distance_to_nearest_pivot(price, pivot, r1, s1, r2, s2)
        }
    }


def _find_nearest_pivot_level(price, pivot, r1, s1, r2=None, s2=None):
    """Helper to find nearest pivot level."""
    levels = {'pivot': pivot, 'r1': r1, 's1': s1}
    if r2: levels['r2'] = r2
    if s2: levels['s2'] = s2
    
    nearest = min(levels.items(), key=lambda x: abs(price - x[1]) if x[1] else float('inf'))
    return nearest[0]


def _distance_to_nearest_pivot(price, pivot, r1, s1, r2=None, s2=None):
    """Helper to find distance to nearest pivot level."""
    levels = [pivot, r1, s1]
    if r2: levels.append(r2)
    if s2: levels.append(s2)
    
    distances = [abs(price - level) / level if level else float('inf') for level in levels if level]
    return min(distances) if distances else 0


@strategy(
    name='fibonacci_retracement',
    feature_discovery=lambda params: [
        FeatureSpec('fibonacci_retracement', {'lookback': params.get('period', 50)}, 'trend'),
        FeatureSpec('fibonacci_retracement', {'lookback': params.get('period', 50)}, '0.382'),
        FeatureSpec('fibonacci_retracement', {'lookback': params.get('period', 50)}, '0.618')
    ],
    parameter_space={
        'period': {'type': 'int', 'range': (10, 100), 'default': 50}
    }
)
def fibonacci_retracement(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Fibonacci retracement level strategy.
    
    Returns sustained signal based on price position relative to Fibonacci zones:
    - Uptrend: 1 above 38.2%, -1 below 61.8%, 0 between
    - Downtrend: -1 below 61.8%, 1 above 38.2%, 0 between
    - No trend: 0
    """
    fib_period = params.get('period', 50)
    
    # Get Fibonacci features (they are decomposed with 'fib_' prefix and shortened levels)
    fib_0 = features.get(f'fibonacci_retracement_{fib_period}_fib_0')      # 0% (high)
    fib_382 = features.get(f'fibonacci_retracement_{fib_period}_fib_38')   # 38.2%
    fib_618 = features.get(f'fibonacci_retracement_{fib_period}_fib_61')   # 61.8%
    fib_100 = features.get(f'fibonacci_retracement_{fib_period}_fib_100')  # 100% (low)
    
    price = bar.get('close', 0)
    
    if fib_382 is None or fib_618 is None or fib_0 is None or fib_100 is None:
        return None
    
    # Determine trend direction from Fibonacci levels (fib_0 is high, fib_100 is low)
    trend_direction = 1 if fib_0 > fib_100 else -1  # Simple trend determination
    
    # Determine signal based on Fibonacci zones and trend
    signal_value = 0
    
    if trend_direction == 1:  # Uptrend (fib_0 > fib_100)
        if price > fib_382:
            signal_value = 1   # Above 38.2% - bullish continuation
        elif price < fib_618:
            signal_value = -1  # Below 61.8% - potential reversal
        else:
            signal_value = 0   # Between 38.2% and 61.8% - neutral zone
    else:  # Downtrend (fib_0 < fib_100)
        if price < fib_618:
            signal_value = -1  # Below 61.8% - bearish continuation
        elif price > fib_382:
            signal_value = 1   # Above 38.2% - potential reversal
        else:
            signal_value = 0   # Between levels - neutral zone
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'fibonacci_retracement',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'period': fib_period,                     # Parameters for sparse storage separation
            'price': price,                           # Values for analysis
            'fib_382': fib_382,
            'fib_618': fib_618,
            'trend_direction': trend_direction if trend_direction is not None else 0,
            'zone': 'above_382' if price > fib_382 else 'below_618' if price < fib_618 else 'neutral'
        }
    }


@strategy(
    name='support_resistance_breakout',
    feature_discovery=lambda params: [
        FeatureSpec('support_resistance', {'lookback': params.get('period', 20)}, 'resistance'),
        FeatureSpec('support_resistance', {'lookback': params.get('period', 20)}, 'support')
    ],
    parameter_space={
        'period': {'type': 'int', 'range': (10, 100), 'default': 20},
        'threshold': {'type': 'float', 'range': (0.001, 0.05), 'default': 0.02}
    }
)
def support_resistance_breakout(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Support/Resistance breakout strategy.
    
    Returns sustained signal based on S/R breakouts:
    - 1: Price breaks above resistance
    - -1: Price breaks below support
    - 0: Price between support and resistance
    """
    sr_period = params.get('period', 20)
    sr_threshold = params.get('threshold', 0.02)
    
    # Get features
    resistance = features.get(f'support_resistance_{sr_period}_resistance')
    support = features.get(f'support_resistance_{sr_period}_support')
    price = bar.get('close', 0)
    
    if resistance is None or support is None:
        return None
    
    # Determine signal based on breakouts
    if price > resistance:
        signal_value = 1   # Resistance breakout
    elif price < support:
        signal_value = -1  # Support breakdown
    else:
        signal_value = 0   # Between levels
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'support_resistance_breakout',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'period': sr_period,                      # Parameters for sparse storage separation
            'threshold': sr_threshold,
            'price': price,                           # Values for analysis
            'resistance': resistance,
            'support': support,
            'range': resistance - support,
            'position_in_range': (price - support) / (resistance - support) if resistance != support else 0.5
        }
    }


@strategy(
    name='atr_channel_breakout',
    feature_discovery=lambda params: [
        FeatureSpec('atr', {'period': params.get('atr_period', 14)}),
        FeatureSpec('sma', {'period': params.get('channel_period', 20)})
    ],
    parameter_space={
        'atr_multiplier': {'type': 'float', 'range': (1.0, 4.0), 'default': 2.0},
        'atr_period': {'type': 'int', 'range': (7, 30), 'default': 14},
        'channel_period': {'type': 'int', 'range': (10, 50), 'default': 20}
    }
)
def atr_channel_breakout(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    ATR channel breakout strategy.
    
    Returns sustained signal based on ATR-based dynamic channels:
    - 1: Price breaks above upper ATR channel
    - -1: Price breaks below lower ATR channel
    - 0: Price within ATR channel
    """
    atr_period = params.get('atr_period', 14)
    channel_period = params.get('channel_period', 20)
    atr_multiplier = params.get('atr_multiplier', 2.0)
    
    # Get features
    atr = features.get(f'atr_{atr_period}')
    middle = features.get(f'sma_{channel_period}')
    price = bar.get('close', 0)
    
    if atr is None or middle is None:
        return None
    
    # Calculate ATR channels
    upper_channel = middle + (atr * atr_multiplier)
    lower_channel = middle - (atr * atr_multiplier)
    
    # Determine signal based on channel position
    if price > upper_channel:
        signal_value = 1   # Breakout above
    elif price < lower_channel:
        signal_value = -1  # Breakdown below
    else:
        signal_value = 0   # Within channel
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'atr_channel_breakout',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'atr_period': atr_period,                 # Parameters for sparse storage separation
            'channel_period': channel_period,
            'atr_multiplier': atr_multiplier,
            'price': price,                           # Values for analysis
            'upper_channel': upper_channel,
            'lower_channel': lower_channel,
            'middle': middle,
            'atr': atr,
            'channel_width': atr * atr_multiplier * 2
        }
    }


@strategy(
    name='price_action_swing',
    feature_discovery=lambda params: [
        FeatureSpec('swing_points', {'lookback': params.get('period', 5)}, 'high'),
        FeatureSpec('swing_points', {'lookback': params.get('period', 5)}, 'low')
    ],
    parameter_space={
        'period': {'type': 'int', 'range': (3, 20), 'default': 10}
    }
)
def price_action_swing(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Price action swing high/low strategy.
    
    Returns sustained signal based on swing points:
    - 1: Higher highs and higher lows (uptrend)
    - -1: Lower highs and lower lows (downtrend)
    - 0: Mixed or no clear pattern
    """
    swing_period = params.get('period', 10)
    
    # Get swing points features (they are decomposed with correct parameter order)
    swing_high = features.get(f'swing_points_{swing_period}_swing_high')
    swing_low = features.get(f'swing_points_{swing_period}_swing_low')
    prev_swing_high = features.get(f'swing_points_{swing_period}_prev_swing_high')
    prev_swing_low = features.get(f'swing_points_{swing_period}_prev_swing_low')
    
    if swing_high is None or swing_low is None:
        return None
    
    # Determine trend based on swing points - only activate when we have historical data
    signal_value = 0
    
    if prev_swing_high is not None and prev_swing_low is not None:
        higher_high = swing_high > prev_swing_high
        higher_low = swing_low > prev_swing_low
        lower_high = swing_high < prev_swing_high
        lower_low = swing_low < prev_swing_low
        
        if higher_high and higher_low:
            signal_value = 1   # Uptrend
        elif lower_high and lower_low:
            signal_value = -1  # Downtrend
        else:
            signal_value = 0   # Mixed/ranging
    # If we don't have prev_swing data yet, strategy stays inactive (returns 0)
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'price_action_swing',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'period': swing_period,                   # Parameters for sparse storage separation
            'swing_high': swing_high,                 # Values for analysis
            'swing_low': swing_low,
            'prev_swing_high': prev_swing_high if prev_swing_high is not None else swing_high,
            'prev_swing_low': prev_swing_low if prev_swing_low is not None else swing_low,
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='swing_pivot_breakout',
    feature_discovery=lambda params: [
        FeatureSpec('swing_points', {'lookback': params.get('swing_period', 10)}, 'high'),
        FeatureSpec('swing_points', {'lookback': params.get('swing_period', 10)}, 'low'),
        FeatureSpec('sma', {'period': 10}),  # For channel basis
        FeatureSpec('atr', {'period': 10})   # For channel width
    ],
    parameter_space={
        'swing_period': {'type': 'int', 'range': (5, 20), 'default': 10},
        'channel_multiplier': {'type': 'float', 'range': (0.5, 3.0), 'default': 1.0},
        'lookback': {'type': 'int', 'range': (20, 60), 'default': 40}  # How far back to look for pivots
    },
    strategy_type='trend_following',
    tags=['trend_following', 'breakout', 'swing', 'channels']
)
def swing_pivot_breakout(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Swing pivot channel breakout strategy (based on Pine Script logic).
    
    Creates dynamic channels from swing highs/lows with volatility-based width.
    Returns sustained signal based on channel breaks:
    - 1: Price breaks above upper channel (bullish breakout)
    - -1: Price breaks below lower channel (bearish breakdown)
    - 0: Price within channels
    
    Channels are set when swing pivots are detected:
    - On swing high: lower channel = basis - deviation
    - On swing low: upper channel = basis + deviation
    """
    swing_period = params.get('swing_period', 10)
    channel_multiplier = params.get('channel_multiplier', 1.0)
    
    # Get features
    swing_high = features.get(f'swing_points_{swing_period}_swing_high')
    swing_low = features.get(f'swing_points_{swing_period}_swing_low')
    basis = features.get('sma_10')
    atr = features.get('atr_10')
    
    price = bar.get('close', 0)
    high = bar.get('high', price)
    low = bar.get('low', price)
    
    if swing_high is None or swing_low is None or basis is None or atr is None:
        return None
    
    # Calculate channel deviation (similar to Pine Script's ta.rma(high - low, 10))
    deviation = atr * channel_multiplier
    
    # Determine channel levels based on most recent swing
    # In real implementation, we'd track these levels across bars
    # For now, using current basis +/- deviation
    upper_channel = basis + deviation
    lower_channel = basis - deviation
    
    # Detect breakouts
    signal_value = 0
    if low > upper_channel:  # Bullish breakout (crossover in Pine Script)
        signal_value = 1
    elif high < lower_channel:  # Bearish breakdown (crossunder in Pine Script)
        signal_value = -1
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'swing_pivot_breakout',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'swing_period': swing_period,
            'channel_multiplier': channel_multiplier,
            'price': price,
            'swing_high': swing_high,
            'swing_low': swing_low,
            'upper_channel': upper_channel,
            'lower_channel': lower_channel,
            'basis': basis,
            'deviation': deviation,
            'channel_width': deviation * 2
        }
    }


@strategy(
    name='swing_pivot_bounce',
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
        'bounce_threshold': {'type': 'float', 'range': (0.001, 0.01), 'default': 0.002},
        'min_touches': {'type': 'int', 'range': (2, 5), 'default': 2},
        'sr_period': {'type': 'int', 'range': (10, 100), 'default': 20},
        'exit_threshold': {'type': 'float', 'range': (0.0, 0.005), 'default': 0.001}  # Distance from midpoint to exit
    },
    strategy_type='mean_reversion',
    tags=['mean_reversion', 'support_resistance', 'bounce', 'swing', 'levels']
)
def swing_pivot_bounce(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Dynamic support/resistance bounce strategy (mean reversion).
    
    Returns sustained signal based on level bounces:
    - 1: Price bounces from support level (expect move up)
    - -1: Price bounces from resistance level (expect move down)
    - 0: No bounce detected
    
    Note: Uses dynamic S/R levels, not pivot points.
    """
    sr_period = params.get('sr_period', 20)
    min_touches = params.get('min_touches', 2)
    bounce_threshold = params.get('bounce_threshold', 0.002)  # 0.2% proximity for bounce
    exit_threshold = params.get('exit_threshold', 0.001)  # 0.1% from midpoint to exit
    
    # Get support/resistance features (they are decomposed by FeatureHub)
    # Feature names include both parameters sorted alphabetically: lookback, min_touches
    resistance = features.get(f'support_resistance_{sr_period}_{min_touches}_resistance')
    support = features.get(f'support_resistance_{sr_period}_{min_touches}_support')
    
    price = bar.get('close', 0)
    high = bar.get('high', price)
    low = bar.get('low', price)
    
    if resistance is None and support is None:
        return None
    
    # Calculate midpoint between support and resistance
    midpoint = (support + resistance) / 2 if support and resistance else None
    
    # Get previous signal to maintain state
    # In a real implementation, this would come from strategy state management
    # For now, we'll use a simplified approach based on current price position
    
    # Determine signal based on position and exit logic
    signal_value = 0
    
    # First, check if we should exit at midpoint
    if midpoint:
        if abs(price - midpoint) / midpoint <= exit_threshold:
            # Price is within exit threshold of midpoint - exit any position
            signal_value = 0
        else:
            # Determine position based on which side of midpoint we're on
            if price < midpoint:
                # Below midpoint - check for long entry at support
                if support and (low <= support * (1 + bounce_threshold) or 
                               abs(price - support) / support < bounce_threshold / 2):
                    signal_value = 1  # Long signal
                else:
                    signal_value = 0  # No signal
            else:
                # Above midpoint - check for short entry at resistance
                if resistance and (high >= resistance * (1 - bounce_threshold) or
                                 abs(price - resistance) / resistance < bounce_threshold / 2):
                    signal_value = -1  # Short signal
                else:
                    signal_value = 0  # No signal
    else:
        # No midpoint available (only one level) - use original logic
        if support and (low <= support * (1 + bounce_threshold) or
                       abs(price - support) / support < bounce_threshold / 2):
            signal_value = 1
        elif resistance and (high >= resistance * (1 - bounce_threshold) or
                           abs(price - resistance) / resistance < bounce_threshold / 2):
            signal_value = -1
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'swing_pivot_bounce',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'sr_period': sr_period,                   # Parameters for sparse storage separation
            'min_touches': min_touches,
            'bounce_threshold': bounce_threshold,
            'exit_threshold': exit_threshold,
            'price': price,                           # Values for analysis
            'resistance': resistance,
            'support': support,
            'midpoint': midpoint,
            'support_distance': abs(price - support) / support if support else None,
            'resistance_distance': abs(price - resistance) / resistance if resistance else None,
            'midpoint_distance': abs(price - midpoint) / midpoint if midpoint else None
        }
    }


@strategy(
    name='trendline_breaks',
    feature_discovery=lambda params: [
        FeatureSpec('trendlines', {
            'pivot_lookback': params.get('pivot_lookback', 20),
            'min_touches': params.get('min_touches', 2),
            'tolerance': params.get('tolerance', 0.002)
        }, 'nearest_resistance'),
        FeatureSpec('trendlines', {
            'pivot_lookback': params.get('pivot_lookback', 20),
            'min_touches': params.get('min_touches', 2),
            'tolerance': params.get('tolerance', 0.002)
        }, 'nearest_support'),
        FeatureSpec('trendlines', {
            'pivot_lookback': params.get('pivot_lookback', 20),
            'min_touches': params.get('min_touches', 2),
            'tolerance': params.get('tolerance', 0.002)
        }, 'valid_uptrends'),
        FeatureSpec('trendlines', {
            'pivot_lookback': params.get('pivot_lookback', 20),
            'min_touches': params.get('min_touches', 2),
            'tolerance': params.get('tolerance', 0.002)
        }, 'valid_downtrends')
    ],
    parameter_space={
        'min_touches': {'type': 'int', 'range': (2, 5), 'default': 2},
        'pivot_lookback': {'type': 'int', 'range': (10, 50), 'default': 20},
        'tolerance': {'type': 'float', 'range': (0.001, 0.005), 'default': 0.002}
    }
)
def trendline_breaks(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Trendline breakout strategy.
    
    Returns sustained signal based on trendline breaks:
    - 1: Price breaks above downtrend line (resistance)
    - -1: Price breaks below uptrend line (support)
    - 0: No break or within trendlines
    """
    pivot_lookback = params.get('pivot_lookback', 20)
    min_touches = params.get('min_touches', 2)
    tolerance = params.get('tolerance', 0.002)
    
    # Get trendline features (using simple names from FeatureHub)
    valid_uptrends = features.get('trendlines_valid_uptrends', 0)
    valid_downtrends = features.get('trendlines_valid_downtrends', 0)
    nearest_support = features.get('trendlines_nearest_support')
    nearest_resistance = features.get('trendlines_nearest_resistance')
    
    if nearest_support is None and nearest_resistance is None:
        return None
    
    price = bar.get('close', 0)
    
    # Determine signal based on trendline breaks
    signal_value = 0
    
    # Breakout signals - fixed logic to match Pine Script
    # Break above resistance (downtrend line) is bullish
    if nearest_resistance and valid_downtrends >= min_touches:
        if price > nearest_resistance * (1 + tolerance):
            signal_value = 1  # Bullish breakout above downtrend
    
    # Break below support (uptrend line) is bearish  
    if nearest_support and valid_uptrends >= min_touches:
        if price < nearest_support * (1 - tolerance):
            signal_value = -1  # Bearish breakdown below uptrend
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'trendline_breaks',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'pivot_lookback': pivot_lookback,         # Parameters for sparse storage separation
            'min_touches': min_touches,
            'tolerance': tolerance,
            'price': price,                           # Values for analysis
            'valid_uptrends': valid_uptrends,
            'valid_downtrends': valid_downtrends,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'breakout_type': 'resistance' if signal_value == 1 else 'support' if signal_value == -1 else 'none'
        }
    }


@strategy(
    name='trendline_bounces',
    feature_discovery=lambda params: [
        FeatureSpec('trendlines', {
            'pivot_lookback': params.get('pivot_lookback', 20),
            'min_touches': params.get('min_touches', 2),
            'tolerance': params.get('tolerance', 0.002)
        }, 'nearest_resistance'),
        FeatureSpec('trendlines', {
            'pivot_lookback': params.get('pivot_lookback', 20),
            'min_touches': params.get('min_touches', 2),
            'tolerance': params.get('tolerance', 0.002)
        }, 'nearest_support'),
        FeatureSpec('trendlines', {
            'pivot_lookback': params.get('pivot_lookback', 20),
            'min_touches': params.get('min_touches', 2),
            'tolerance': params.get('tolerance', 0.002)
        }, 'strongest_uptrend'),
        FeatureSpec('trendlines', {
            'pivot_lookback': params.get('pivot_lookback', 20),
            'min_touches': params.get('min_touches', 2),
            'tolerance': params.get('tolerance', 0.002)
        }, 'strongest_downtrend')
    ],
    parameter_space={
        'bounce_threshold': {'type': 'float', 'range': (0.001, 0.01), 'default': 0.003},
        'min_touches': {'type': 'int', 'range': (2, 5), 'default': 3},
        'min_bounces': {'type': 'int', 'range': (0, 5), 'default': 0},  # Minimum successful bounces
        'pivot_lookback': {'type': 'int', 'range': (10, 50), 'default': 20},
        'tolerance': {'type': 'float', 'range': (0.001, 0.005), 'default': 0.002}
    },
    strategy_type='mean_reversion'
)
def trendline_bounces(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Trendline bounce strategy (mean reversion).
    
    Returns sustained signal based on trendline bounces:
    - 1: Price bounces from uptrend line (support)
    - -1: Price bounces from downtrend line (resistance)
    - 0: No bounce or weak signal
    
    Can optionally require minimum successful bounces before trading.
    """
    pivot_lookback = params.get('pivot_lookback', 20)
    min_touches = params.get('min_touches', 3)
    tolerance = params.get('tolerance', 0.002)
    bounce_threshold = params.get('bounce_threshold', 0.003)  # 0.3% proximity
    min_bounces = params.get('min_bounces', 0)  # Minimum successful bounces before trading
    
    # Get trendline features (using simple names from FeatureHub)
    nearest_support = features.get('trendlines_nearest_support')
    nearest_resistance = features.get('trendlines_nearest_resistance')
    valid_uptrends = features.get('trendlines_valid_uptrends', 0)
    valid_downtrends = features.get('trendlines_valid_downtrends', 0)
    support_bounces = features.get('trendlines_support_bounces', 0)
    resistance_bounces = features.get('trendlines_resistance_bounces', 0)
    
    if nearest_support is None and nearest_resistance is None:
        return None
    
    price = bar.get('close', 0)
    low = bar.get('low', price)
    high = bar.get('high', price)
    
    # Determine signal based on proximity to validated trendlines
    signal_value = 0
    
    # Bounce from uptrend support line (bullish)
    if nearest_support and valid_uptrends >= min_touches and support_bounces >= min_bounces:
        # Check if low touched support and closed above it
        if low <= nearest_support * (1 + bounce_threshold) and price > nearest_support:
            signal_value = 1  # Bounce from support (long)
        # Or very close to support (anticipatory)
        elif abs(price - nearest_support) / nearest_support < bounce_threshold / 2:
            signal_value = 1
    
    # Bounce from downtrend resistance line (bearish)
    elif nearest_resistance and valid_downtrends >= min_touches and resistance_bounces >= min_bounces:
        # Check if high touched resistance and closed below it
        if high >= nearest_resistance * (1 - bounce_threshold) and price < nearest_resistance:
            signal_value = -1  # Bounce from resistance (short)
        # Or very close to resistance (anticipatory)
        elif abs(price - nearest_resistance) / nearest_resistance < bounce_threshold / 2:
            signal_value = -1
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'trendline_bounces',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'pivot_lookback': pivot_lookback,         # Parameters for sparse storage separation
            'min_touches': min_touches,
            'tolerance': tolerance,
            'bounce_threshold': bounce_threshold,
            'price': price,                           # Values for analysis
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'valid_uptrends': valid_uptrends,
            'valid_downtrends': valid_downtrends,
            'support_bounces': support_bounces,
            'resistance_bounces': resistance_bounces,
            'support_distance': abs(price - nearest_support) / nearest_support if nearest_support else None,
            'resistance_distance': abs(price - nearest_resistance) / nearest_resistance if nearest_resistance else None,
            'bounce_type': 'support' if signal_value == 1 else 'resistance' if signal_value == -1 else 'none'
        }
    }


def _diagonal_channel_features(params):
    """Helper to create diagonal channel feature specs without repetition."""
    channel_params = {
        'lookback': params.get('lookback', 20),
        'min_points': params.get('min_points', 3),
        'channel_tolerance': params.get('channel_tolerance', 0.02),
        'parallel_tolerance': params.get('parallel_tolerance', 0.1)
    }
    # Return single spec to get all outputs
    return [FeatureSpec('diagonal_channel', channel_params)]


@strategy(
    name='diagonal_channel_reversion',
    feature_discovery=_diagonal_channel_features,
    parameter_space={
        'lookback': {'type': 'int', 'range': (10, 50), 'default': 20},
        'min_points': {'type': 'int', 'range': (2, 5), 'default': 3},
        'min_bounces': {'type': 'int', 'range': (0, 5), 'default': 0},
        'channel_tolerance': {'type': 'float', 'range': (0.01, 0.05), 'default': 0.02},
        'parallel_tolerance': {'type': 'float', 'range': (0.05, 0.2), 'default': 0.1},
        
        # Entry configuration
        'entry_mode': {'type': 'categorical', 'choices': ['boundary', 'midline', 'both'], 'default': 'boundary'},
        'boundary_threshold': {'type': 'float', 'range': (0.001, 0.01), 'default': 0.003},
        'midline_distance': {'type': 'float', 'range': (0.1, 0.4), 'default': 0.25},
        
        # Target configuration
        'target_mode': {'type': 'categorical', 'choices': ['opposite', 'midline', 'percent'], 'default': 'midline'},
        'target_percent': {'type': 'float', 'range': (0.2, 0.8), 'default': 0.5}
    },
    strategy_type='mean_reversion',
    tags=['mean_reversion', 'channels', 'diagonal', 'configurable']
)
def diagonal_channel_reversion(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Configurable diagonal channel reversion strategy.
    
    Entry modes:
    - 'boundary': Enter at channel boundaries
    - 'midline': Enter when far from midline
    - 'both': Combine both conditions
    
    Target modes:
    - 'opposite': Target opposite boundary
    - 'midline': Target channel midline
    - 'percent': Target percentage of channel width
    """
    # Get parameters
    entry_mode = params.get('entry_mode', 'boundary')
    boundary_threshold = params.get('boundary_threshold', 0.003)
    midline_distance = params.get('midline_distance', 0.25)
    target_mode = params.get('target_mode', 'midline')
    target_percent = params.get('target_percent', 0.5)
    min_bounces = params.get('min_bounces', 0)
    
    # Get all channel features
    upper = features.get('diagonal_channel_upper_channel')
    lower = features.get('diagonal_channel_lower_channel')
    mid = features.get('diagonal_channel_mid_channel')
    position = features.get('diagonal_channel_position_in_channel')
    upper_bounces = features.get('diagonal_channel_upper_bounces', 0)
    lower_bounces = features.get('diagonal_channel_lower_bounces', 0)
    
    if not all([upper, lower, mid, position is not None]):
        return None
    
    price = bar.get('close', 0)
    high = bar.get('high', price)
    low = bar.get('low', price)
    
    signal_value = 0
    entry_reason = None
    
    # Entry logic
    if entry_mode in ['boundary', 'both']:
        # Boundary entries
        if lower_bounces >= min_bounces and position < boundary_threshold:
            signal_value = 1
            entry_reason = 'lower_boundary'
        elif upper_bounces >= min_bounces and position > 1 - boundary_threshold:
            signal_value = -1
            entry_reason = 'upper_boundary'
    
    if entry_mode in ['midline', 'both'] and signal_value == 0:
        # Midline deviation entries
        if position < 0.5 - midline_distance:
            signal_value = 1
            entry_reason = 'below_midline'
        elif position > 0.5 + midline_distance:
            signal_value = -1
            entry_reason = 'above_midline'
    
    # Calculate target
    target_price = None
    if signal_value != 0:
        width = upper - lower
        if target_mode == 'opposite':
            target_price = upper if signal_value == 1 else lower
        elif target_mode == 'midline':
            target_price = mid
        elif target_mode == 'percent':
            target_price = price + (width * target_percent * signal_value)
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'diagonal_channel_reversion',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'price': price,
            'upper_channel': upper,
            'lower_channel': lower,
            'mid_channel': mid,
            'position': position,
            'entry_mode': entry_mode,
            'entry_reason': entry_reason,
            'target_mode': target_mode,
            'target_price': target_price,
            'channel_angle': features.get('diagonal_channel_channel_angle', 0),
            'channel_strength': features.get('diagonal_channel_channel_strength', 0)
        }
    }


@strategy(
    name='diagonal_channel_breakout',
    feature_discovery=_diagonal_channel_features,
    parameter_space={
        'lookback': {'type': 'int', 'range': (10, 50), 'default': 20},
        'min_points': {'type': 'int', 'range': (2, 5), 'default': 3},
        'channel_tolerance': {'type': 'float', 'range': (0.01, 0.05), 'default': 0.02},
        'parallel_tolerance': {'type': 'float', 'range': (0.05, 0.2), 'default': 0.1},
        'breakout_threshold': {'type': 'float', 'range': (0.001, 0.01), 'default': 0.002},
        'exit_mode': {'type': 'categorical', 'choices': ['channel_touch', 'opposite_touch', 'midline'], 'default': 'channel_touch'}
    },
    strategy_type='trend_following',
    tags=['trend_following', 'breakout', 'channels', 'diagonal']
)
def diagonal_channel_breakout(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Diagonal channel breakout strategy with sloped exit logic.
    
    Entry: Price breaks out of channel
    Exit modes:
    - 'channel_touch': Exit when price returns to breakout channel line
    - 'opposite_touch': Exit only at opposite channel line
    - 'midline': Exit when price crosses midline
    
    The channel lines naturally trail due to their slope, providing
    a form of trailing exit that locks in profits.
    """
    breakout_threshold = params.get('breakout_threshold', 0.002)
    exit_mode = params.get('exit_mode', 'channel_touch')
    
    # Get channel features
    upper = features.get('diagonal_channel_upper_channel')
    lower = features.get('diagonal_channel_lower_channel')
    mid = features.get('diagonal_channel_mid_channel')
    position = features.get('diagonal_channel_position_in_channel')
    is_current = features.get('diagonal_channel_channel_is_current', True)
    
    if upper is None or lower is None:
        return None
    
    price = bar.get('close', 0)
    
    # Breakout detection
    above_upper = price > upper * (1 + breakout_threshold)
    below_lower = price < lower * (1 - breakout_threshold)
    
    # Exit detection based on mode
    signal_value = 0
    exit_reason = None
    
    if exit_mode == 'channel_touch':
        # Exit when price touches the breakout line again
        if above_upper:
            signal_value = 1  # Stay long while above upper
        elif below_lower:
            signal_value = -1  # Stay short while below lower
        # Signal becomes 0 when price is back inside channel
        
    elif exit_mode == 'opposite_touch':
        # Hold until opposite channel is touched
        if above_upper and price > lower:
            signal_value = 1
        elif below_lower and price < upper:
            signal_value = -1
            
    elif exit_mode == 'midline':
        # Exit at midline cross
        if above_upper and price > mid:
            signal_value = 1
        elif below_lower and price < mid:
            signal_value = -1
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'diagonal_channel_breakout',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'price': price,
            'upper_channel': upper,
            'lower_channel': lower,
            'mid_channel': mid,
            'position': position,
            'channel_angle': features.get('diagonal_channel_channel_angle', 0),
            'channel_is_current': is_current,
            'exit_mode': exit_mode,
            'above_upper': above_upper,
            'below_lower': below_lower
        }
    }