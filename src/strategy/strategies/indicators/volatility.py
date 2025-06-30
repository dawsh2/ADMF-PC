"""
Volatility-based indicator strategies.

All volatility strategies that generate signals based on volatility measures
and channel breakouts/penetrations.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='keltner_breakout',
    feature_discovery=lambda params: [
        FeatureSpec('keltner_channel', {
            'period': params.get('period', 20),
            'multiplier': params.get('multiplier', 2.0)
        }, 'upper'),
        FeatureSpec('keltner_channel', {
            'period': params.get('period', 20),
            'multiplier': params.get('multiplier', 2.0)
        }, 'middle'),
        FeatureSpec('keltner_channel', {
            'period': params.get('period', 20),
            'multiplier': params.get('multiplier', 2.0)
        }, 'lower')
    ],
    parameter_space={
        'multiplier': {'type': 'float', 'range': (1.0, 4.0), 'default': 2.0},
        'period': {'type': 'int', 'range': (10, 50), 'default': 20}
    },
    strategy_type='trend_following',
    tags=['breakout', 'volatility', 'keltner', 'trend_following']
)
def keltner_breakout(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Keltner Channel breakout strategy (trend-following).
    
    Returns sustained signal based on price breaking Keltner Channel:
    - 1: Price > upper band (bullish breakout)
    - -1: Price < lower band (bearish breakout)
    - 0: Price within channel
    """
    period = params.get('period', 20)
    multiplier = params.get('multiplier', 2.0)
    
    # Get features - parameters are sorted alphabetically in canonical names
    # So multiplier comes before period: keltner_channel_{multiplier}_{period}
    upper_band = features.get(f'keltner_channel_{multiplier}_{period}_upper')
    lower_band = features.get(f'keltner_channel_{multiplier}_{period}_lower')
    middle_band = features.get(f'keltner_channel_{multiplier}_{period}_middle')
    price = bar.get('close', 0)
    
    if upper_band is None or lower_band is None:
        return None
    
    # Breakout logic (trend-following)
    if price > upper_band:
        signal_value = 1   # Bullish breakout - BUY
    elif price < lower_band:
        signal_value = -1  # Bearish breakout - SELL
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
    },
    parameter_space={
        'period': {'type': 'int', 'range': (10, 50), 'default': 20}
    },
    strategy_type='trend_following',
    tags=['breakout', 'volatility', 'donchian', 'trend_following', 'turtle']
)
def donchian_breakout(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Donchian Channel breakout strategy (trend-following).
    
    Returns sustained signal based on price breaking Donchian Channel:
    - 1: Price > upper band (new high breakout - bullish)
    - -1: Price < lower band (new low breakout - bearish)
    - 0: Price within channel
    
    Classic turtle trading style breakout.
    """
    period = params.get('period', 20)
    
    # Get features
    upper_band = features.get(f'donchian_channel_{period}_upper')
    lower_band = features.get(f'donchian_channel_{period}_lower')
    middle_band = features.get(f'donchian_channel_{period}_middle')
    price = bar.get('close', 0)
    
    if upper_band is None or lower_band is None:
        return None
    
    # Breakout logic (trend-following)
    if price > upper_band:
        signal_value = 1   # New high breakout - BUY
    elif price < lower_band:
        signal_value = -1  # New low breakout - SELL
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
    name='bollinger_bands',
    feature_discovery=lambda params: [
        FeatureSpec('bollinger_bands', {
            'period': params.get('period', 20),
            'std_dev': params.get('std_dev', 2.0)
        }, 'upper'),
        FeatureSpec('bollinger_bands', {
            'period': params.get('period', 20),
            'std_dev': params.get('std_dev', 2.0)
        }, 'middle'),
        FeatureSpec('bollinger_bands', {
            'period': params.get('period', 20),
            'std_dev': params.get('std_dev', 2.0)
        }, 'lower')
    ],
    parameter_space={
        'period': {'type': 'int', 'range': (10, 50), 'default': 20},
        'std_dev': {'type': 'float', 'range': (1.5, 3.0), 'default': 2.0},  # Band width control
        'exit_threshold': {'type': 'float', 'range': (0.0, 0.005), 'default': 0.001}  # Exit when within 0.1% of middle
    },
    strategy_type='mean_reversion',
    tags=['mean_reversion', 'volatility', 'bollinger', 'simple']
)
def bollinger_bands(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Bollinger Bands mean reversion strategy with middle band exit.
    
    Entry signals:
    - 1: Price at or below lower band (oversold - buy)
    - -1: Price at or above upper band (overbought - sell)
    
    Exit signals:
    - 0: Price within exit_threshold of middle band
    
    Band width is controlled by the std_dev parameter.
    """
    period = params.get('period', 20)
    std_dev = params.get('std_dev', 2.0)
    exit_threshold = params.get('exit_threshold', 0.001)  # 0.1% default
    
    # Get features
    upper_band = features.get(f'bollinger_bands_{period}_{std_dev}_upper')
    lower_band = features.get(f'bollinger_bands_{period}_{std_dev}_lower')
    middle_band = features.get(f'bollinger_bands_{period}_{std_dev}_middle')
    price = bar.get('close', 0)
    
    if upper_band is None or lower_band is None or middle_band is None:
        return None
    
    # First check exit condition - within threshold of middle band
    # COMMENTED OUT: This causes excessive trading by exiting at middle band
    # if abs(price - middle_band) / middle_band <= exit_threshold:
    #     signal_value = 0  # Exit any position
    # Then check entry conditions
    if price >= upper_band:
        signal_value = -1  # Short at upper band
    elif price <= lower_band:
        signal_value = 1   # Long at lower band
    else:
        signal_value = 0   # Neutral - not at entry or exit levels
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'bollinger_bands',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'price': price,
            'open': bar.get('open', 0),
            'high': bar.get('high', 0),
            'low': bar.get('low', 0),
            'close': bar.get('close', 0),
            'upper_band': upper_band,
            'lower_band': lower_band,
            'middle_band': middle_band,
            'band_width': upper_band - lower_band,
            'band_position': (price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
        }
    }


@strategy(
    name='bollinger_breakout',
    feature_discovery=lambda params: [
        FeatureSpec('bollinger_bands', {
            'period': params.get('period', 20),
            'std_dev': params.get('std_dev', 2.0)
        }, 'upper'),
        FeatureSpec('bollinger_bands', {
            'period': params.get('period', 20),
            'std_dev': params.get('std_dev', 2.0)
        }, 'middle'),
        FeatureSpec('bollinger_bands', {
            'period': params.get('period', 20),
            'std_dev': params.get('std_dev', 2.0)
        }, 'lower')
    ],
    parameter_space={
        'period': {'type': 'int', 'range': (10, 50), 'default': 20},
        'std_dev': {'type': 'float', 'range': (1.0, 3.0), 'default': 2.0}
    },
    strategy_type='trend_following',
    tags=['breakout', 'volatility', 'bollinger']
)
def bollinger_breakout(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Bollinger Bands breakout strategy (trend-following).
    
    Returns sustained signal based on price breaking Bollinger Bands:
    - 1: Price breaks above upper band (bullish breakout)
    - -1: Price breaks below lower band (bearish breakout)
    - 0: Price within bands
    
    This is a trend-following strategy, opposite of mean reversion.
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
    
    # Breakout logic (trend-following)
    if price > upper_band:
        signal_value = 1   # Bullish breakout - BUY
    elif price < lower_band:
        signal_value = -1  # Bearish breakout - SELL
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
    name='keltner_bands',
    feature_discovery=lambda params: [
        FeatureSpec('keltner_channel', {
            'period': params.get('period', 20),
            'multiplier': params.get('multiplier', 2.0)
        }, 'upper'),
        FeatureSpec('keltner_channel', {
            'period': params.get('period', 20),
            'multiplier': params.get('multiplier', 2.0)
        }, 'middle'),
        FeatureSpec('keltner_channel', {
            'period': params.get('period', 20),
            'multiplier': params.get('multiplier', 2.0)
        }, 'lower')
    ],
    parameter_space={
        'multiplier': {'type': 'float', 'range': (1.0, 4.0), 'default': 2.0},
        'period': {'type': 'int', 'range': (10, 50), 'default': 20}
    },
    strategy_type='mean_reversion',
    tags=['mean_reversion', 'volatility', 'keltner']
)
def keltner_bands(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Keltner Channel mean reversion strategy.
    
    Returns sustained signal based on price vs Keltner Channel:
    - 1: Price <= lower band (oversold - buy)
    - -1: Price >= upper band (overbought - sell)
    - 0: Price within channel
    """
    period = params.get('period', 20)
    multiplier = params.get('multiplier', 2.0)
    
    # Get features - parameters are sorted alphabetically in canonical names
    # So multiplier comes before period: keltner_channel_{multiplier}_{period}
    upper_band = features.get(f'keltner_channel_{multiplier}_{period}_upper')
    lower_band = features.get(f'keltner_channel_{multiplier}_{period}_lower')
    middle_band = features.get(f'keltner_channel_{multiplier}_{period}_middle')
    price = bar.get('close', 0)
    
    if upper_band is None or lower_band is None:
        return None
    
    # Mean reversion logic
    if price >= upper_band:
        signal_value = -1  # Overbought - sell
    elif price <= lower_band:
        signal_value = 1   # Oversold - buy
    else:
        signal_value = 0   # Within channel
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'keltner_bands',
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
    name='donchian_bands',
    feature_discovery=lambda params: [
        FeatureSpec('donchian_channel', {'period': params.get('period', 20)})
    ],
    parameter_space={
        'period': {'type': 'int', 'range': (10, 50), 'default': 20},
        'buffer_pct': {'type': 'float', 'range': (0.0, 0.01), 'default': 0.002},  # 0-1% buffer
        'use_middle_exit': {'type': 'bool', 'default': False}  # Exit at middle band
    },
    strategy_type='mean_reversion',
    tags=['mean_reversion', 'volatility', 'donchian']
)
def donchian_bands(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Donchian Channel mean reversion strategy.
    
    Returns sustained signal based on price vs Donchian Channel:
    - 1: Price <= lower band (at period low - buy)
    - -1: Price >= upper band (at period high - sell)
    - 0: Price within channel
    
    Trades extremes expecting reversion to mean.
    """
    period = params.get('period', 20)
    buffer_pct = params.get('buffer_pct', 0.002)  # 0.2% default buffer
    use_middle_exit = params.get('use_middle_exit', False)
    
    # Get features
    upper_band = features.get(f'donchian_channel_{period}_upper')
    lower_band = features.get(f'donchian_channel_{period}_lower')
    middle_band = features.get(f'donchian_channel_{period}_middle')
    price = bar.get('close', 0)
    
    if upper_band is None or lower_band is None:
        return None
    
    # Apply buffer to bands to prevent immediate reversal
    upper_trigger = upper_band * (1 - buffer_pct)  # Slightly below upper band
    lower_trigger = lower_band * (1 + buffer_pct)  # Slightly above lower band
    
    # Mean reversion logic with buffer
    if price >= upper_trigger:
        signal_value = -1  # Near high - expect reversion down
    elif price <= lower_trigger:
        signal_value = 1   # Near low - expect reversion up
    elif use_middle_exit and abs(price - middle_band) / middle_band < 0.001:
        signal_value = 0   # Exit near middle
    else:
        # Maintain previous signal until opposite band is reached
        # This prevents flipping every bar
        signal_value = 0   # Neutral when in middle of channel
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'donchian_bands',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'price': price,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'middle_band': middle_band,
            'channel_width': upper_band - lower_band
        }
    }


