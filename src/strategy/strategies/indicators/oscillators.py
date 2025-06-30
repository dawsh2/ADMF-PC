"""
Oscillator-based indicator strategies.

All oscillator strategies that generate signals based on bounded indicators
that oscillate between fixed ranges (typically 0-100).
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='rsi_threshold',
    feature_discovery=lambda params: [
        FeatureSpec('rsi', {'period': params.get('rsi_period', 14)})
    ],
    parameter_space={
        'rsi_period': {'type': 'int', 'range': (7, 30), 'default': 14},
        'threshold': {'type': 'float', 'range': (0, 100), 'default': 50}
    }
)
def rsi_threshold(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    RSI threshold strategy.
    
    Returns sustained signal based on RSI vs threshold:
    - 1: RSI > threshold (bullish momentum)
    - -1: RSI < threshold (bearish momentum)
    - 0: RSI at threshold
    """
    rsi_period = params.get('rsi_period', 14)
    threshold = params.get('threshold', 50)
    
    # Get features
    rsi = features.get(f'rsi_{rsi_period}')
    
    if rsi is None:
        return None
    
    # Determine signal based on threshold
    if rsi > threshold:
        signal_value = 1  # Above threshold
    elif rsi < threshold:
        signal_value = -1  # Below threshold
    else:
        signal_value = 0  # At threshold
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'rsi_threshold',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'rsi_period': rsi_period,          # Parameters for sparse storage separation
            'threshold': threshold,
            'rsi': rsi,                        # Values for analysis
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='rsi_bands',
    feature_discovery=lambda params: [
        FeatureSpec('rsi', {'period': params.get('rsi_period', 14)})
    ],
    parameter_space={
        'overbought': {'type': 'float', 'range': (60, 90), 'default': 70, 'granularity': 4},
        'oversold': {'type': 'float', 'range': (10, 40), 'default': 30, 'granularity': 4},
        'rsi_period': {'type': 'int', 'range': (7, 30), 'default': 14, 'granularity': 3}
    },
    strategy_type='mean_reversion',
    tags=['mean_reversion', 'oscillator', 'rsi', 'overbought_oversold']
)
def rsi_bands(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    RSI overbought/oversold bands strategy.
    
    Returns sustained signal based on RSI bands:
    - -1: RSI > overbought (overbought - mean reversion signal)
    - 1: RSI < oversold (oversold - mean reversion signal)
    - 0: RSI in neutral zone
    """
    rsi_period = params.get('rsi_period', 14)
    overbought = params.get('overbought', 70)
    oversold = params.get('oversold', 30)
    
    # Get features
    rsi = features.get(f'rsi_{rsi_period}')
    
    if rsi is None:
        return None
    
    # Determine signal based on bands
    if rsi > overbought:
        signal_value = -1  # Overbought (mean reversion signal)
    elif rsi < oversold:
        signal_value = 1   # Oversold (mean reversion signal)
    else:
        signal_value = 0   # Neutral zone
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'rsi_bands',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'rsi_period': rsi_period,          # Parameters for sparse storage separation
            'overbought': overbought,
            'oversold': oversold,
            'rsi': rsi,                        # Values for analysis
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='cci_threshold',
    feature_discovery=lambda params: [
        FeatureSpec('cci', {'period': params.get('cci_period', 20)})
    ],
    parameter_space={
        'cci_period': {'type': 'int', 'range': (5, 100), 'default': 20},
        'threshold': {'type': 'float', 'range': (-200, 200), 'default': 0}
    }
)
def cci_threshold(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    CCI threshold strategy.
    
    Returns sustained signal based on CCI vs threshold:
    - 1: CCI > threshold (bullish momentum)
    - -1: CCI < threshold (bearish momentum)
    - 0: CCI at threshold
    """
    cci_period = params.get('cci_period', 20)
    threshold = params.get('threshold', 0)
    
    # Get features
    cci = features.get(f'cci_{cci_period}')
    
    if cci is None:
        return None
    
    # Determine signal based on threshold
    if cci > threshold:
        signal_value = 1  # Above threshold
    elif cci < threshold:
        signal_value = -1  # Below threshold
    else:
        signal_value = 0  # At threshold
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'cci_threshold',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'cci_period': cci_period,          # Parameters for sparse storage separation
            'threshold': threshold,
            'cci': cci,                        # Values for analysis
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='cci_bands',
    feature_discovery=lambda params: [
        FeatureSpec('cci', {'period': params.get('cci_period', 20)})
    ],
    parameter_space={
        'cci_period': {'type': 'int', 'range': (5, 100), 'default': 20},
        'overbought': {'type': 'float', 'range': (50, 200), 'default': 100},
        'oversold': {'type': 'float', 'range': (-200, -50), 'default': -100}
    },
    strategy_type='mean_reversion'
)
def cci_bands(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    CCI extreme bands strategy.
    
    Returns sustained signal based on CCI extreme levels:
    - -1: CCI > overbought (extreme overbought - mean reversion signal)
    - 1: CCI < oversold (extreme oversold - mean reversion signal)
    - 0: CCI in normal range
    """
    cci_period = params.get('cci_period', 20)
    overbought = params.get('overbought', 100)
    oversold = params.get('oversold', -100)
    
    # Get features
    cci = features.get(f'cci_{cci_period}')
    
    if cci is None:
        return None
    
    # Determine signal based on bands
    if cci > overbought:
        signal_value = -1  # Overbought (mean reversion signal)
    elif cci < oversold:
        signal_value = 1   # Oversold (mean reversion signal)
    else:
        signal_value = 0   # Neutral zone
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'cci_bands',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'cci_period': cci_period,          # Parameters for sparse storage separation
            'overbought': overbought,
            'oversold': oversold,
            'cci': cci,                        # Values for analysis
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='stochastic_rsi',
    feature_discovery=lambda params: [
        FeatureSpec('stochastic_rsi', {
            'rsi_period': params.get('rsi_period', 14),
            'stoch_period': params.get('stoch_period', 14),
            'd_period': params.get('d_period', 3)
        })
    ],
    parameter_space={
        'overbought': {'type': 'float', 'range': (60, 90), 'default': 80},
        'oversold': {'type': 'float', 'range': (10, 40), 'default': 20},
        'rsi_period': {'type': 'int', 'range': (7, 30), 'default': 14},
        'stoch_period': {'type': 'int', 'range': (5, 30), 'default': 14},
        'd_period': {'type': 'int', 'range': (1, 10), 'default': 3}
    },
    strategy_type='mean_reversion'
)
def stochastic_rsi(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Stochastic RSI oscillator strategy.
    
    Returns sustained signal based on StochRSI extreme levels:
    - -1: StochRSI > overbought (mean reversion short)
    - 1: StochRSI < oversold (mean reversion long)
    - 0: StochRSI in neutral zone
    """
    rsi_period = params.get('rsi_period', 14)
    stoch_period = params.get('stoch_period', 14)
    overbought = params.get('overbought', 80)
    oversold = params.get('oversold', 20)
    
    # Get features
    stoch_rsi_k = features.get(f'stochastic_rsi_{rsi_period}_{stoch_period}_k')
    stoch_rsi_d = features.get(f'stochastic_rsi_{rsi_period}_{stoch_period}_d')
    
    # Use K line for signal, D line for smoothing
    stoch_rsi = stoch_rsi_k if stoch_rsi_k is not None else stoch_rsi_d
    
    if stoch_rsi is None:
        return None
    
    # Determine signal based on extreme levels
    if stoch_rsi > overbought:
        signal_value = -1  # Overbought (mean reversion)
    elif stoch_rsi < oversold:
        signal_value = 1   # Oversold (mean reversion)
    else:
        signal_value = 0   # Neutral zone
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'stochastic_rsi',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'rsi_period': rsi_period,          # Parameters for sparse storage separation
            'stoch_period': stoch_period,
            'overbought': overbought,
            'oversold': oversold,
            'stoch_rsi_k': stoch_rsi_k if stoch_rsi_k is not None else stoch_rsi,  # Values for analysis
            'stoch_rsi_d': stoch_rsi_d if stoch_rsi_d is not None else stoch_rsi,
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='williams_r',
    feature_discovery=lambda params: [
        FeatureSpec('williams_r', {'period': params.get('williams_period', 14)})
    ],
    parameter_space={
        'overbought': {'type': 'float', 'range': (-40, 0), 'default': -20},
        'oversold': {'type': 'float', 'range': (-100, -60), 'default': -80},
        'williams_period': {'type': 'int', 'range': (5, 50), 'default': 14}
    },
    strategy_type='mean_reversion'
)
def williams_r(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Williams %R oscillator strategy.
    
    Returns sustained signal based on Williams %R levels:
    - -1: %R > overbought (closer to 0, mean reversion short)
    - 1: %R < oversold (closer to -100, mean reversion long)
    - 0: %R in neutral zone
    
    Note: Williams %R is inverted (0 to -100) compared to other oscillators
    """
    williams_period = params.get('williams_period', 14)
    overbought = params.get('overbought', -20)  # Closer to 0
    oversold = params.get('oversold', -80)      # Closer to -100
    
    # Get features
    williams = features.get(f'williams_r_{williams_period}')
    
    if williams is None:
        return None
    
    # Determine signal (note inverted logic due to Williams %R scale)
    if williams > overbought:
        signal_value = -1  # Overbought (mean reversion)
    elif williams < oversold:
        signal_value = 1   # Oversold (mean reversion)
    else:
        signal_value = 0   # Neutral zone
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'williams_r',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'williams_period': williams_period,  # Parameters for sparse storage separation
            'overbought': overbought,
            'oversold': oversold,
            'williams_r': williams,              # Values for analysis
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='roc_threshold',
    feature_discovery=lambda params: [
        FeatureSpec('roc', {'period': params.get('roc_period', 10)})
    ],
    parameter_space={
        'roc_period': {'type': 'int', 'range': (5, 50), 'default': 10},
        'threshold': {'type': 'float', 'range': (0.5, 10.0), 'default': 2.0}
    }
)
def roc_threshold(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Rate of Change (ROC) threshold strategy.
    
    Returns sustained signal based on ROC vs threshold:
    - 1: ROC > threshold (bullish momentum)
    - -1: ROC < -threshold (bearish momentum)
    - 0: ROC between thresholds (neutral)
    """
    roc_period = params.get('roc_period', 10)
    threshold = params.get('threshold', 2.0)  # 2% default
    
    # Get features
    roc = features.get(f'roc_{roc_period}')
    
    if roc is None:
        return None
    
    # Determine signal based on threshold
    if roc > threshold:
        signal_value = 1   # Bullish momentum
    elif roc < -threshold:
        signal_value = -1  # Bearish momentum
    else:
        signal_value = 0   # Neutral
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'roc_threshold',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'roc_period': roc_period,          # Parameters for sparse storage separation
            'threshold': threshold,
            'roc': roc,                        # Values for analysis
            'price': bar.get('close', 0),
            'momentum': 'bullish' if roc > threshold else 'bearish' if roc < -threshold else 'neutral'
        }
    }


@strategy(
    name='ultimate_oscillator',
    feature_discovery=lambda params: [
        FeatureSpec('ultimate_oscillator', {
            'period1': params.get('period1', 7),
            'period2': params.get('period2', 14),
            'period3': params.get('period3', 28)
        })
    ],
    parameter_space={
        'overbought': {'type': 'float', 'range': (60, 90), 'default': 70},
        'oversold': {'type': 'float', 'range': (10, 40), 'default': 30},
        'period1': {'type': 'int', 'range': (5, 15), 'default': 7},
        'period2': {'type': 'int', 'range': (10, 30), 'default': 14},
        'period3': {'type': 'int', 'range': (20, 50), 'default': 28}
    },
    strategy_type='mean_reversion'
)
def ultimate_oscillator(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Ultimate Oscillator strategy.
    
    Returns sustained signal based on UO extreme levels:
    - -1: UO > overbought (mean reversion short)
    - 1: UO < oversold (mean reversion long)
    - 0: UO in neutral zone
    """
    uo_period1 = params.get('period1', 7)
    uo_period2 = params.get('period2', 14)
    uo_period3 = params.get('period3', 28)
    overbought = params.get('overbought', 70)
    oversold = params.get('oversold', 30)
    
    # Get features
    uo = features.get(f'ultimate_oscillator_{uo_period1}_{uo_period2}_{uo_period3}')
    
    if uo is None:
        return None
    
    # Determine signal based on extreme levels
    if uo > overbought:
        signal_value = -1  # Overbought (mean reversion)
    elif uo < oversold:
        signal_value = 1   # Oversold (mean reversion)
    else:
        signal_value = 0   # Neutral zone
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'ultimate_oscillator',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'period1': uo_period1,             # Parameters for sparse storage separation
            'period2': uo_period2,
            'period3': uo_period3,
            'overbought': overbought,
            'oversold': oversold,
            'ultimate_oscillator': uo,         # Values for analysis
            'price': bar.get('close', 0)
        }
    }