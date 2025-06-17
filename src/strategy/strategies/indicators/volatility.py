"""
Volatility-based indicator strategies.

All volatility strategies that generate signals based on volatility measures
and channel breakouts/penetrations.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy


@strategy(
    name='keltner_breakout',
    feature_config=['keltner_channel'],  # Simple: just declare we need Keltner Channel features
    param_feature_mapping={
        'period': 'keltner_channel_{period}_{multiplier}',
        'multiplier': 'keltner_channel_{period}_{multiplier}'
    }
)
def keltner_breakout(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Keltner Channel breakout strategy.
    
    Returns sustained signal based on price vs Keltner Channel:
    - -1: Price > upper band (mean reversion short signal)
    - 1: Price < lower band (mean reversion long signal)
    - 0: Price within channel
    """
    period = params.get('period', 20)
    multiplier = params.get('multiplier', 2.0)
    
    # Get features
    upper_band = features.get(f'keltner_channel_{period}_{multiplier}_upper')
    lower_band = features.get(f'keltner_channel_{period}_{multiplier}_lower')
    middle_band = features.get(f'keltner_channel_{period}_{multiplier}_middle')
    price = bar.get('close', 0)
    
    if upper_band is None or lower_band is None:
        return None
    
    # Determine signal based on channel position (mean reversion logic)
    if price > upper_band:
        signal_value = -1  # Mean reversion short
    elif price < lower_band:
        signal_value = 1   # Mean reversion long
    else:
        signal_value = 0   # Within channel
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'keltner_breakout',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'price': price,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'middle_band': middle_band
        }
    }


@strategy(
    name='donchian_breakout',
    feature_config=['donchian_channel'],  # Simple: just declare we need Donchian Channel features
    param_feature_mapping={
        'period': 'donchian_channel_{period}'
    }
)
def donchian_breakout(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Donchian Channel breakout strategy.
    
    Returns sustained signal based on price vs Donchian Channel:
    - -1: Price > upper band (mean reversion short signal)
    - 1: Price < lower band (mean reversion long signal)
    - 0: Price within channel
    """
    period = params.get('period', 20)
    
    # Get features
    upper_band = features.get(f'donchian_channel_{period}_upper')
    lower_band = features.get(f'donchian_channel_{period}_lower')
    middle_band = features.get(f'donchian_channel_{period}_middle')
    price = bar.get('close', 0)
    
    if upper_band is None or lower_band is None:
        return None
    
    # Determine signal based on channel position (mean reversion logic)
    if price > upper_band:
        signal_value = -1  # Mean reversion short
    elif price < lower_band:
        signal_value = 1   # Mean reversion long
    else:
        signal_value = 0   # Within channel
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'donchian_breakout',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'price': price,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'middle_band': middle_band
        }
    }


@strategy(
    name='bollinger_breakout',
    feature_config=['bollinger_bands'],  # Simple: just declare we need Bollinger Bands features
    param_feature_mapping={
        'period': 'bollinger_bands_{period}_{std_dev}',
        'std_dev': 'bollinger_bands_{period}_{std_dev}'
    }
)
def bollinger_breakout(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Bollinger Bands breakout/reversion strategy.
    
    Returns sustained signal based on price vs Bollinger Bands.
    Can operate in breakout mode or mean reversion mode.
    """
    period = params.get('period', 20)
    std_dev = params.get('std_dev', 2.0)
    # Get features
    upper_band = features.get(f'bollinger_bands_{period}_{std_dev}_upper')
    lower_band = features.get(f'bollinger_bands_{period}_{std_dev}_lower')
    middle_band = features.get(f'bollinger_bands_{period}_{std_dev}_middle')
    price = bar.get('close', 0)
    
    if upper_band is None or lower_band is None:
        return None
    
    # Mean reversion logic (matching reference implementation)
    if price > upper_band:
        signal_value = -1  # Mean reversion short
    elif price < lower_band:
        signal_value = 1   # Mean reversion long
    else:
        signal_value = 0   # Within bands
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'bollinger_breakout',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'price': price,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'middle_band': middle_band,
            'band_width': upper_band - lower_band
        }
    }


@strategy(
    name='bollinger_mean_reversion',
    feature_config=['bollinger_bands', 'rsi'],  # Use RSI for additional filtering
    param_feature_mapping={
        'bb_period': 'bollinger_bands_{bb_period}_{bb_std}',
        'bb_std': 'bollinger_bands_{bb_period}_{bb_std}',
        'rsi_period': 'rsi_{rsi_period}'
    }
)
def bollinger_mean_reversion(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Bollinger Bands mean reversion strategy with RSI filter.
    
    Trades bounces from the bands back to the middle:
    - 1: Price at/below lower band AND RSI oversold → Long (expect bounce up)
    - -1: Price at/above upper band AND RSI overbought → Short (expect pullback)
    - 0: Otherwise or exit when price returns to middle band
    """
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    rsi_period = params.get('rsi_period', 14)
    rsi_oversold = params.get('rsi_oversold', 30)
    rsi_overbought = params.get('rsi_overbought', 70)
    
    # Get features
    upper_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_upper')
    lower_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_lower')
    middle_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_middle')
    rsi = features.get(f'rsi_{rsi_period}')
    price = bar.get('close', 0)
    
    if any(f is None for f in [upper_band, lower_band, middle_band, rsi]):
        return None
    
    # Calculate band position (0 = lower band, 1 = upper band)
    band_position = (price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
    
    # Mean reversion signals with RSI confirmation
    signal_value = 0
    
    # Long signal: Price at lower band + RSI oversold
    if band_position <= 0.1 and rsi < rsi_oversold:
        signal_value = 1
    # Short signal: Price at upper band + RSI overbought  
    elif band_position >= 0.9 and rsi > rsi_overbought:
        signal_value = -1
    # Exit signals: Price returned to middle band (within 10% of middle)
    elif abs(price - middle_band) / middle_band < 0.001:
        signal_value = 0
    
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'bollinger_mean_reversion',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'price': price,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'middle_band': middle_band,
            'band_position': band_position,
            'rsi': rsi,
            'band_width': upper_band - lower_band,
            'price_to_middle': (price - middle_band) / middle_band * 100
        }
    }


@strategy(
    name='keltner_mean_reversion',
    feature_config=['keltner_channel', 'atr'],  # Use ATR for volatility context
    param_feature_mapping={
        'kc_period': 'keltner_channel_{kc_period}_{kc_mult}',
        'kc_mult': 'keltner_channel_{kc_period}_{kc_mult}',
        'atr_period': 'atr_{atr_period}'
    }
)
def keltner_mean_reversion(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Keltner Channel mean reversion strategy with volatility filter.
    
    Trades bounces from the channels back to the middle:
    - 1: Price touches lower channel → Long (expect bounce up)
    - -1: Price touches upper channel → Short (expect pullback)
    - 0: Price near middle or volatility too high
    
    Unlike Bollinger Bands, Keltner Channels use ATR for width, making them
    more responsive to volatility changes.
    """
    kc_period = params.get('kc_period', 20)
    kc_mult = params.get('kc_mult', 2.0)
    atr_period = params.get('atr_period', 14)
    max_atr_pct = params.get('max_atr_pct', 2.0)  # Max ATR as % of price
    
    # Get features
    upper_channel = features.get(f'keltner_channel_{kc_period}_{kc_mult}_upper')
    lower_channel = features.get(f'keltner_channel_{kc_period}_{kc_mult}_lower')
    middle_channel = features.get(f'keltner_channel_{kc_period}_{kc_mult}_middle')
    atr = features.get(f'atr_{atr_period}')
    price = bar.get('close', 0)
    
    if any(f is None for f in [upper_channel, lower_channel, middle_channel, atr]):
        return None
    
    # Calculate channel position and volatility
    channel_width = upper_channel - lower_channel
    channel_position = (price - lower_channel) / channel_width if channel_width > 0 else 0.5
    atr_pct = (atr / price) * 100 if price > 0 else 0
    
    # Skip signals in high volatility
    if atr_pct > max_atr_pct:
        signal_value = 0
    else:
        # Mean reversion signals
        if channel_position <= 0.05:  # Price at/below lower channel
            signal_value = 1
        elif channel_position >= 0.95:  # Price at/above upper channel
            signal_value = -1
        else:
            signal_value = 0
    
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'keltner_mean_reversion',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'price': price,
            'upper_channel': upper_channel,
            'lower_channel': lower_channel,
            'middle_channel': middle_channel,
            'channel_position': channel_position,
            'channel_width': channel_width,
            'atr': atr,
            'atr_pct': atr_pct,
            'price_to_middle': (price - middle_channel) / middle_channel * 100
        }
    }