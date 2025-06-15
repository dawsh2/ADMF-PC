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