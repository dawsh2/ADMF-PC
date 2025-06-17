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
    name='pivot_channel_breaks',
    feature_config=['pivot_points', 'support_resistance'],  # Use existing features
    param_feature_mapping={
        'pivot_type': 'pivot_points_{pivot_type}',
        'sr_period': 'support_resistance_{sr_period}'
    }
)
def pivot_channel_breaks(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Pivot Channel breakout strategy using pivot points and support/resistance.
    
    Returns sustained signal based on level breaks:
    - 1: Price breaks above R1 or resistance level
    - -1: Price breaks below S1 or support level
    - 0: Price within levels
    """
    pivot_type = params.get('pivot_type', 'standard')
    sr_period = params.get('sr_period', 20)
    breakout_threshold = params.get('breakout_threshold', 0.001)  # 0.1% threshold
    
    # Get pivot point features (they are decomposed by FeatureHub)
    r1 = features.get(f'pivot_points_{pivot_type}_r1')
    s1 = features.get(f'pivot_points_{pivot_type}_s1')
    pivot = features.get(f'pivot_points_{pivot_type}_pivot')
    
    # Get support/resistance features (they are decomposed by FeatureHub)
    resistance = features.get(f'support_resistance_{sr_period}_resistance')
    support = features.get(f'support_resistance_{sr_period}_support')
    
    price = bar.get('close', 0)
    
    if r1 is None or s1 is None:
        return None
    
    if r1 is None or s1 is None:
        return None
    
    # Determine signal based on level breaks
    signal_value = 0
    upper_level = max(r1, resistance or r1)
    lower_level = min(s1, support or s1)
    
    # Apply breakout threshold
    if price > upper_level * (1 + breakout_threshold):
        signal_value = 1   # Bullish breakout
    elif price < lower_level * (1 - breakout_threshold):
        signal_value = -1  # Bearish breakdown
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'pivot_channel_breaks',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'pivot_type': pivot_type,                 # Parameters for sparse storage separation
            'sr_period': sr_period,
            'breakout_threshold': breakout_threshold,
            'price': price,                           # Values for analysis
            'r1': r1,
            's1': s1,
            'resistance': resistance,
            'support': support,
            'upper_level': upper_level,
            'lower_level': lower_level,
            'channel_width': upper_level - lower_level
        }
    }


@strategy(
    name='pivot_channel_bounces',
    feature_config=['support_resistance'],  # Use existing support/resistance feature
    param_feature_mapping={
        'sr_period': 'support_resistance_{sr_period}'
    }
)
def pivot_channel_bounces(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Support/Resistance bounce strategy.
    
    Returns sustained signal based on level bounces:
    - 1: Price bounces from support level (mean reversion long)
    - -1: Price bounces from resistance level (mean reversion short)
    - 0: No bounce detected
    """
    sr_period = params.get('sr_period', 20)
    min_touches = params.get('min_touches', 2)
    bounce_threshold = params.get('bounce_threshold', 0.002)  # 0.2% proximity for bounce
    
    # Get support/resistance features (they are decomposed by FeatureHub)
    resistance = features.get(f'support_resistance_{sr_period}_resistance')
    support = features.get(f'support_resistance_{sr_period}_support')
    
    price = bar.get('close', 0)
    high = bar.get('high', price)
    low = bar.get('low', price)
    
    if resistance is None and support is None:
        return None
    
    # Determine signal based on proximity to levels
    signal_value = 0
    
    # Check for bounce from support (mean reversion long)
    if support and low <= support * (1 + bounce_threshold) and price > support:
        # Price touched support and bounced up
        signal_value = 1
    
    # Check for bounce from resistance (mean reversion short)  
    elif resistance and high >= resistance * (1 - bounce_threshold) and price < resistance:
        # Price touched resistance and bounced down
        signal_value = -1
    
    # Additional proximity-based signals
    elif support and abs(price - support) / support < bounce_threshold / 2:
        # Very close to support - anticipatory long
        signal_value = 1
    elif resistance and abs(price - resistance) / resistance < bounce_threshold / 2:
        # Very close to resistance - anticipatory short
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
            'sr_period': sr_period,                   # Parameters for sparse storage separation
            'min_touches': min_touches,
            'bounce_threshold': bounce_threshold,
            'price': price,                           # Values for analysis
            'resistance': resistance,
            'support': support,
            'support_distance': abs(price - support) / support if support else None,
            'resistance_distance': abs(price - resistance) / resistance if resistance else None
        }
    }


@strategy(
    name='trendline_breaks',
    feature_config=['trendlines'],  # Simple: just declare we need trendline features
    param_feature_mapping={}  # Use simple feature names without parameter mapping
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
    feature_config=['trendlines'],  # Simple: just declare we need trendline features
    param_feature_mapping={}  # Use simple feature names without parameter mapping
)
def trendline_bounces(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Trendline bounce strategy (mean reversion).
    
    Returns sustained signal based on trendline bounces:
    - 1: Price bounces from uptrend line (support)
    - -1: Price bounces from downtrend line (resistance)
    - 0: No bounce or weak signal
    """
    pivot_lookback = params.get('pivot_lookback', 20)
    min_touches = params.get('min_touches', 3)
    tolerance = params.get('tolerance', 0.002)
    bounce_threshold = params.get('bounce_threshold', 0.003)  # 0.3% proximity
    
    # Get trendline features (using simple names from FeatureHub)
    nearest_support = features.get('trendlines_nearest_support')
    nearest_resistance = features.get('trendlines_nearest_resistance')
    valid_uptrends = features.get('trendlines_valid_uptrends', 0)
    valid_downtrends = features.get('trendlines_valid_downtrends', 0)
    
    if nearest_support is None and nearest_resistance is None:
        return None
    
    price = bar.get('close', 0)
    low = bar.get('low', price)
    high = bar.get('high', price)
    
    # Determine signal based on proximity to validated trendlines
    signal_value = 0
    
    # Bounce from uptrend support line (bullish)
    if nearest_support and valid_uptrends >= min_touches:
        # Check if low touched support and closed above it
        if low <= nearest_support * (1 + bounce_threshold) and price > nearest_support:
            signal_value = 1  # Bounce from support (long)
        # Or very close to support (anticipatory)
        elif abs(price - nearest_support) / nearest_support < bounce_threshold / 2:
            signal_value = 1
    
    # Bounce from downtrend resistance line (bearish)
    elif nearest_resistance and valid_downtrends >= min_touches:
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
            'support_distance': abs(price - nearest_support) / nearest_support if nearest_support else None,
            'resistance_distance': abs(price - nearest_resistance) / nearest_resistance if nearest_resistance else None,
            'bounce_type': 'support' if signal_value == 1 else 'resistance' if signal_value == -1 else 'none'
        }
    }