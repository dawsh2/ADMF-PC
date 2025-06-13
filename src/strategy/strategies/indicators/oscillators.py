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
    - -1: RSI > upper_threshold (overbought - mean reversion signal)
    - 1: RSI < lower_threshold (oversold - mean reversion signal)
    - 0: RSI in neutral zone
    """
    rsi_period = params.get('rsi_period', 14)
    upper_threshold = params.get('upper_threshold', 70)
    lower_threshold = params.get('lower_threshold', 30)
    
    # Get features
    rsi = features.get(f'rsi_{rsi_period}')
    
    if rsi is None:
        return None
    
    # Determine signal based on bands
    if rsi > upper_threshold:
        signal_value = -1  # Overbought (mean reversion signal)
    elif rsi < lower_threshold:
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
            'upper_threshold': upper_threshold,
            'lower_threshold': lower_threshold,
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
    - -1: CCI > upper_threshold (extreme overbought - mean reversion signal)
    - 1: CCI < lower_threshold (extreme oversold - mean reversion signal)
    - 0: CCI in normal range
    """
    cci_period = params.get('cci_period', 20)
    upper_threshold = params.get('upper_threshold', 100)
    lower_threshold = params.get('lower_threshold', -100)
    
    # Get features
    cci = features.get(f'cci_{cci_period}')
    
    if cci is None:
        return None
    
    # Determine signal based on bands
    if cci > upper_threshold:
        signal_value = -1  # Overbought (mean reversion signal)
    elif cci < lower_threshold:
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
            'upper_threshold': upper_threshold,
            'lower_threshold': lower_threshold,
            'price': bar.get('close', 0)
        }
    }