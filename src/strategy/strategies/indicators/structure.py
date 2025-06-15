"""
Market structure indicator strategies.

All structure strategies that generate signals based on support/resistance,
pivot points, and price patterns.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy


@strategy(
    name='pivot_points',
    feature_config=['pivot_points'],  # Simple: just declare we need pivot point features
    param_feature_mapping={
        'pivot_type': 'pivot_points_{pivot_type}'
    }
)
def pivot_points(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Pivot point support/resistance strategy.
    
    Returns sustained signal based on price vs pivot levels:
    - 1: Price breaks above R1 (bullish breakout)
    - -1: Price breaks below S1 (bearish breakdown)
    - 0: Price between S1 and R1 (ranging)
    """
    pivot_type = params.get('pivot_type', 'standard')
    
    # Get features
    pivot = features.get(f'pivot_points_{pivot_type}_pivot')
    r1 = features.get(f'pivot_points_{pivot_type}_r1')
    s1 = features.get(f'pivot_points_{pivot_type}_s1')
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
            'pivot_type': pivot_type,                 # Parameters for sparse storage separation
            'price': price,                           # Values for analysis
            'pivot': pivot,
            'r1': r1,
            's1': s1,
            'position': 'above_r1' if price > r1 else 'below_s1' if price < s1 else 'neutral'
        }
    }


@strategy(
    name='fibonacci_retracement',
    feature_config=['fibonacci_retracement'],  # Simple: just declare we need Fibonacci features
    param_feature_mapping={
        'period': 'fibonacci_retracement_{period}'
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
    
    # Get features
    fib_0 = features.get(f'fibonacci_retracement_{fib_period}_0')      # 0% (high)
    fib_236 = features.get(f'fibonacci_retracement_{fib_period}_236')  # 23.6%
    fib_382 = features.get(f'fibonacci_retracement_{fib_period}_382')  # 38.2%
    fib_618 = features.get(f'fibonacci_retracement_{fib_period}_618')  # 61.8%
    fib_100 = features.get(f'fibonacci_retracement_{fib_period}_100')  # 100% (low)
    trend_direction = features.get(f'fibonacci_retracement_{fib_period}_trend')  # 1 for up, -1 for down
    price = bar.get('close', 0)
    
    if fib_382 is None or fib_618 is None:
        return None
    
    # Determine signal based on Fibonacci zones and trend
    signal_value = 0
    
    if trend_direction == 1:  # Uptrend
        if price > fib_382:
            signal_value = 1   # Above 38.2% - bullish continuation
        elif price < fib_618:
            signal_value = -1  # Below 61.8% - potential reversal
        else:
            signal_value = 0   # Between 38.2% and 61.8% - neutral zone
    elif trend_direction == -1:  # Downtrend  
        if price < fib_618:
            signal_value = -1  # Below 61.8% - bearish continuation
        elif price > fib_382:
            signal_value = 1   # Above 38.2% - potential reversal
        else:
            signal_value = 0   # Between levels - neutral zone
    else:
        # No clear trend
        signal_value = 0
    
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
    feature_config=['support_resistance'],  # Simple: just declare we need support/resistance features
    param_feature_mapping={
        'period': 'support_resistance_{period}'
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
    feature_config=['atr', 'sma'],  # Simple: declare we need ATR and SMA features
    param_feature_mapping={
        'atr_period': 'atr_{atr_period}',
        'channel_period': 'sma_{channel_period}'
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
    feature_config=['swing_points'],  # Simple: just declare we need swing point features
    param_feature_mapping={
        'period': 'swing_points_{period}'
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
    
    # Get features
    swing_high = features.get(f'swing_points_high_{swing_period}')
    swing_low = features.get(f'swing_points_low_{swing_period}')
    prev_swing_high = features.get(f'swing_points_prev_high_{swing_period}')
    prev_swing_low = features.get(f'swing_points_prev_low_{swing_period}')
    
    if swing_high is None or swing_low is None:
        return None
    
    # Determine trend based on swing points
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
    name='pivot_channel_breaks',
    feature_config=['pivot_channels'],  # Simple: just declare we need pivot channel features
    param_feature_mapping={}
)
def pivot_channel_breaks(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Pivot Channel breakout strategy.
    
    Returns sustained signal based on channel breaks:
    - 1: Price breaks above pivot high or upper channel
    - -1: Price breaks below pivot low or lower channel
    - 0: Price within channels
    """
    channel_feature = 'pivot_channels'
    
    # Get all pivot channel values
    pivot_high = features.get(f'{channel_feature}_pivot_high')
    pivot_low = features.get(f'{channel_feature}_pivot_low')
    upper_channel = features.get(f'{channel_feature}_upper_channel')
    lower_channel = features.get(f'{channel_feature}_lower_channel')
    break_up = features.get(f'{channel_feature}_break_up', False)
    break_down = features.get(f'{channel_feature}_break_down', False)
    
    price = bar.get('close', 0)
    
    if pivot_high is None or pivot_low is None:
        return None
    
    # Determine signal based on breaks
    signal_value = 0
    
    if break_up:
        signal_value = 1   # Bullish breakout
    elif break_down:
        signal_value = -1  # Bearish breakdown
    elif upper_channel and lower_channel:
        # Check position within channels
        if price > upper_channel:
            signal_value = 1
        elif price < lower_channel:
            signal_value = -1
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'pivot_channel_breaks',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'price': price,                           # Values for analysis
            'pivot_high': pivot_high,
            'pivot_low': pivot_low,
            'upper_channel': upper_channel,
            'lower_channel': lower_channel,
            'channel_width': upper_channel - lower_channel if upper_channel and lower_channel else 0
        }
    }


@strategy(
    name='pivot_channel_bounces',
    feature_config=['pivot_channels'],  # Simple: just declare we need pivot channel features
    param_feature_mapping={
        'min_touches': 'pivot_channels_{min_touches}'
    }
)
def pivot_channel_bounces(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Pivot Channel bounce strategy.
    
    Returns sustained signal based on channel bounces:
    - 1: Price bounces from lower channel or pivot low
    - -1: Price bounces from upper channel or pivot high
    - 0: No bounce detected
    """
    channel_feature = 'pivot_channels'
    min_touches = params.get('min_touches', 2)
    
    # Get bounce signals and touch counts
    bounce_up = features.get(f'{channel_feature}_bounce_up', False)
    bounce_down = features.get(f'{channel_feature}_bounce_down', False)
    upper_touches = features.get(f'{channel_feature}_upper_touches', 0)
    lower_touches = features.get(f'{channel_feature}_lower_touches', 0)
    pivot_high_touches = features.get(f'{channel_feature}_pivot_high_touches', 0)
    pivot_low_touches = features.get(f'{channel_feature}_pivot_low_touches', 0)
    
    # Get proximity scores
    proximity_upper = features.get(f'{channel_feature}_proximity_upper', 0)
    proximity_lower = features.get(f'{channel_feature}_proximity_lower', 0)
    
    price = bar.get('close', 0)
    
    # Determine signal based on bounces
    signal_value = 0
    
    # Only trade validated levels (minimum touches)
    if bounce_up and (lower_touches >= min_touches or pivot_low_touches >= min_touches):
        signal_value = 1   # Bullish bounce
    elif bounce_down and (upper_touches >= min_touches or pivot_high_touches >= min_touches):
        signal_value = -1  # Bearish bounce
    elif proximity_lower > 0.8 and lower_touches >= min_touches:
        # Anticipatory long when very close to validated support
        signal_value = 1
    elif proximity_upper > 0.8 and upper_touches >= min_touches:
        # Anticipatory short when very close to validated resistance
        signal_value = -1
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'pivot_channel_bounces',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'min_touches': min_touches,               # Parameters for sparse storage separation
            'price': price,                           # Values for analysis
            'upper_touches': upper_touches,
            'lower_touches': lower_touches,
            'pivot_high_touches': pivot_high_touches,
            'pivot_low_touches': pivot_low_touches,
            'proximity_upper': proximity_upper,
            'proximity_lower': proximity_lower
        }
    }


@strategy(
    name='trendline_breaks',
    feature_config=['trendlines'],  # Simple: just declare we need trendline features
    param_feature_mapping={
        'min_strength': 'trendlines_{min_strength}'
    }
)
def trendline_breaks(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Trendline breakout strategy.
    
    Returns sustained signal based on trendline breaks:
    - 1: Price breaks above downtrend line
    - -1: Price breaks below uptrend line
    - 0: No break or mixed signals
    """
    min_strength = params.get('min_strength', 0.2)
    
    # Get trendline data
    recent_breaks = features.get('trendlines_recent_breaks', [])
    valid_uptrends = features.get('trendlines_valid_uptrends', 0)
    valid_downtrends = features.get('trendlines_valid_downtrends', 0)
    
    price = bar.get('close', 0)
    
    # Determine signal based on recent breaks
    signal_value = 0
    
    if recent_breaks:
        # Check most recent break
        last_break = recent_breaks[-1]
        
        if last_break['type'] == 'downtrend':
            # Breaking above downtrend is bullish
            signal_value = 1
        elif last_break['type'] == 'uptrend':
            # Breaking below uptrend is bearish
            signal_value = -1
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'trendline_breaks',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'min_strength': min_strength,             # Parameters for sparse storage separation
            'price': price,                           # Values for analysis
            'valid_uptrends': valid_uptrends,
            'valid_downtrends': valid_downtrends,
            'break_count': len(recent_breaks),
            'last_break': recent_breaks[-1] if recent_breaks else None
        }
    }


@strategy(
    name='trendline_bounces',
    feature_config=['trendlines'],  # Simple: just declare we need trendline features
    param_feature_mapping={
        'min_touches': 'trendlines_{min_touches}',
        'min_strength': 'trendlines_{min_strength}'
    }
)
def trendline_bounces(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Trendline bounce strategy.
    
    Returns sustained signal based on trendline bounces:
    - 1: Price bounces from uptrend line
    - -1: Price bounces from downtrend line
    - 0: No bounce or weak signal
    """
    min_touches = params.get('min_touches', 3)
    min_strength = params.get('min_strength', 0.3)
    
    # Get trendline data
    recent_bounces = features.get('trendlines_recent_bounces', [])
    nearest_support = features.get('trendlines_nearest_support')
    nearest_resistance = features.get('trendlines_nearest_resistance')
    
    price = bar.get('close', 0)
    
    # Determine signal based on bounces
    signal_value = 0
    
    if recent_bounces:
        # Filter bounces by minimum criteria
        valid_bounces = [b for b in recent_bounces 
                        if b['touches'] >= min_touches and b['strength'] >= min_strength]
        
        if valid_bounces:
            # Use most recent valid bounce
            last_bounce = valid_bounces[-1]
            
            if last_bounce['type'] == 'uptrend':
                signal_value = 1   # Bullish bounce
            elif last_bounce['type'] == 'downtrend':
                signal_value = -1  # Bearish bounce
    
    # Check proximity for anticipatory signals
    elif nearest_support and price > nearest_support:
        support_distance = (price - nearest_support) / price
        if support_distance < 0.002:  # Within 0.2%
            signal_value = 1  # Anticipatory long
    elif nearest_resistance and price < nearest_resistance:
        resistance_distance = (nearest_resistance - price) / price
        if resistance_distance < 0.002:  # Within 0.2%
            signal_value = -1  # Anticipatory short
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'trendline_bounces',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'min_touches': min_touches,               # Parameters for sparse storage separation
            'min_strength': min_strength,
            'price': price,                           # Values for analysis
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'bounce_count': len(recent_bounces),
            'valid_bounces': len([b for b in recent_bounces if b['touches'] >= min_touches])
        }
    }