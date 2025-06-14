"""
Oscillator-based indicator strategies.

All oscillator strategies that generate signals based on bounded indicators
that oscillate between fixed ranges (typically 0-100).
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy


@strategy(
    name='rsi_threshold',
    feature_config={
        'rsi': {
            'params': ['rsi_period'],
            'defaults': {'rsi_period': 14}
        }
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
            'rsi': rsi,
            'threshold': threshold,
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='rsi_bands',
    feature_config={
        'rsi': {
            'params': ['rsi_period'],
            'defaults': {'rsi_period': 14}
        }
    }
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
            'rsi': rsi,
            'overbought': overbought,
            'oversold': oversold,
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='cci_threshold',
    feature_config={
        'cci': {
            'params': ['cci_period'],
            'defaults': {'cci_period': 20}
        }
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
            'cci': cci,
            'threshold': threshold,
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='cci_bands',
    feature_config={
        'cci': {
            'params': ['cci_period'],
            'defaults': {'cci_period': 20}
        }
    }
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
            'cci': cci,
            'overbought': overbought,
            'oversold': oversold,
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='stochastic_rsi',
    feature_config={
        'stochastic_rsi': {
            'params': ['rsi_period', 'stoch_period'],
            'defaults': {'rsi_period': 14, 'stoch_period': 14}
        }
    }
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
    stoch_rsi_k = features.get(f'stoch_rsi_{rsi_period}_{stoch_period}_k')
    stoch_rsi_d = features.get(f'stoch_rsi_{rsi_period}_{stoch_period}_d')
    
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
            'stoch_rsi_k': stoch_rsi_k if stoch_rsi_k is not None else stoch_rsi,
            'stoch_rsi_d': stoch_rsi_d if stoch_rsi_d is not None else stoch_rsi,
            'overbought': overbought,
            'oversold': oversold,
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='williams_r',
    feature_config={
        'williams_r': {
            'params': ['williams_period'],
            'defaults': {'williams_period': 14}
        }
    }
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
            'williams_r': williams,
            'overbought': overbought,
            'oversold': oversold,
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='roc_threshold',
    feature_config={
        'roc': {
            'params': ['roc_period'],
            'defaults': {'roc_period': 10}
        }
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
            'roc': roc,
            'threshold': threshold,
            'price': bar.get('close', 0),
            'momentum': 'bullish' if roc > threshold else 'bearish' if roc < -threshold else 'neutral'
        }
    }


@strategy(
    name='ultimate_oscillator',
    feature_config={
        'ultimate_oscillator': {
            'params': ['uo_period1', 'uo_period2', 'uo_period3'],
            'defaults': {'uo_period1': 7, 'uo_period2': 14, 'uo_period3': 28}
        }
    }
)
def ultimate_oscillator(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Ultimate Oscillator strategy.
    
    Returns sustained signal based on UO extreme levels:
    - -1: UO > overbought (mean reversion short)
    - 1: UO < oversold (mean reversion long)
    - 0: UO in neutral zone
    """
    uo_period1 = params.get('uo_period1', 7)
    uo_period2 = params.get('uo_period2', 14)
    uo_period3 = params.get('uo_period3', 28)
    overbought = params.get('overbought', 70)
    oversold = params.get('oversold', 30)
    
    # Get features
    uo = features.get(f'uo_{uo_period1}_{uo_period2}_{uo_period3}')
    
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
            'ultimate_oscillator': uo,
            'overbought': overbought,
            'oversold': oversold,
            'price': bar.get('close', 0)
        }
    }