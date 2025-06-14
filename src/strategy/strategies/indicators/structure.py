"""
Market structure indicator strategies.

All structure strategies that generate signals based on support/resistance,
pivot points, and price patterns.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy


@strategy(
    name='pivot_points',
    feature_config={
        'pivot': {
            'params': ['pivot_type'],
            'defaults': {'pivot_type': 'standard'}
        }
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
    pivot = features.get(f'pivot_{pivot_type}')
    r1 = features.get(f'pivot_{pivot_type}_r1')
    s1 = features.get(f'pivot_{pivot_type}_s1')
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
            'price': price,
            'pivot': pivot,
            'r1': r1,
            's1': s1,
            'position': 'above_r1' if price > r1 else 'below_s1' if price < s1 else 'neutral'
        }
    }


@strategy(
    name='fibonacci_retracement',
    feature_config={
        'fibonacci': {
            'params': ['fib_period'],
            'defaults': {'fib_period': 50}
        }
    }
)
def fibonacci_retracement(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Fibonacci retracement level strategy.
    
    Returns sustained signal based on price vs Fibonacci levels:
    - 1: Price bounces from 38.2% or 61.8% in uptrend
    - -1: Price rejects from 38.2% or 61.8% in downtrend
    - 0: Price between levels or at extreme (0% or 100%)
    """
    fib_period = params.get('fib_period', 50)
    
    # Get features
    fib_0 = features.get(f'fib_{fib_period}_0')      # 0% (high)
    fib_236 = features.get(f'fib_{fib_period}_236')  # 23.6%
    fib_382 = features.get(f'fib_{fib_period}_382')  # 38.2%
    fib_618 = features.get(f'fib_{fib_period}_618')  # 61.8%
    fib_100 = features.get(f'fib_{fib_period}_100')  # 100% (low)
    trend_direction = features.get(f'fib_{fib_period}_trend')  # 1 for up, -1 for down
    price = bar.get('close', 0)
    
    if fib_382 is None or fib_618 is None:
        return None
    
    # Determine signal based on Fibonacci levels and trend
    signal_value = 0
    bounce_threshold = params.get('bounce_threshold', 0.002)  # 0.2% proximity
    
    # Check if price is near key Fibonacci levels
    near_382 = abs(price - fib_382) / price < bounce_threshold if price != 0 else False
    near_618 = abs(price - fib_618) / price < bounce_threshold if price != 0 else False
    
    if trend_direction == 1:  # Uptrend
        if near_382 or near_618:
            signal_value = 1  # Bounce from retracement
    elif trend_direction == -1:  # Downtrend
        if near_382 or near_618:
            signal_value = -1  # Rejection from retracement
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'fibonacci_retracement',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'price': price,
            'fib_382': fib_382,
            'fib_618': fib_618,
            'trend_direction': trend_direction if trend_direction is not None else 0,
            'near_382': near_382,
            'near_618': near_618
        }
    }


@strategy(
    name='support_resistance_breakout',
    feature_config={
        'support_resistance': {
            'params': ['sr_period', 'sr_threshold'],
            'defaults': {'sr_period': 20, 'sr_threshold': 0.02}
        }
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
    sr_period = params.get('sr_period', 20)
    sr_threshold = params.get('sr_threshold', 0.02)
    
    # Get features
    resistance = features.get(f'resistance_{sr_period}')
    support = features.get(f'support_{sr_period}')
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
            'price': price,
            'resistance': resistance,
            'support': support,
            'range': resistance - support,
            'position_in_range': (price - support) / (resistance - support) if resistance != support else 0.5
        }
    }


@strategy(
    name='atr_channel_breakout',
    feature_config={
        'atr': {
            'params': ['atr_period'],
            'defaults': {'atr_period': 14}
        },
        'sma': {
            'params': ['channel_period'],
            'defaults': {'channel_period': 20}
        }
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
            'price': price,
            'upper_channel': upper_channel,
            'lower_channel': lower_channel,
            'middle': middle,
            'atr': atr,
            'channel_width': atr * atr_multiplier * 2
        }
    }


@strategy(
    name='price_action_swing',
    feature_config={
        'swing': {
            'params': ['swing_period'],
            'defaults': {'swing_period': 10}
        }
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
    swing_period = params.get('swing_period', 10)
    
    # Get features
    swing_high = features.get(f'swing_high_{swing_period}')
    swing_low = features.get(f'swing_low_{swing_period}')
    prev_swing_high = features.get(f'prev_swing_high_{swing_period}')
    prev_swing_low = features.get(f'prev_swing_low_{swing_period}')
    
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
            'swing_high': swing_high,
            'swing_low': swing_low,
            'prev_swing_high': prev_swing_high if prev_swing_high is not None else swing_high,
            'prev_swing_low': prev_swing_low if prev_swing_low is not None else swing_low,
            'price': bar.get('close', 0)
        }
    }