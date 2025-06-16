"""
Volume-based indicator strategies.

All volume strategies that generate signals based on volume analysis
and volume-price relationships.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy


@strategy(
    name='obv_trend',
    feature_config=['obv', 'obv_sma'],  # Need OBV and SMA of OBV
    param_feature_mapping={
        'obv_sma_period': 'obv_sma_{obv_sma_period}'
    }
)
def obv_trend(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    On-Balance Volume trend strategy.
    
    Returns sustained signal based on OBV vs its moving average:
    - 1: OBV > OBV_SMA (accumulation)
    - -1: OBV < OBV_SMA (distribution)
    - 0: Equal
    """
    obv_sma_period = params.get('obv_sma_period', 20)
    
    # Get features
    obv = features.get('obv')
    obv_sma = features.get(f'obv_sma_{obv_sma_period}')
    
    if obv is None or obv_sma is None:
        return None
    
    # Determine signal
    if obv > obv_sma:
        signal_value = 1  # Accumulation
    elif obv < obv_sma:
        signal_value = -1  # Distribution
    else:
        signal_value = 0
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'obv_trend',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'obv_sma_period': obv_sma_period,         # Parameters for sparse storage separation
            'obv': obv,                               # Values for analysis
            'obv_sma': obv_sma,
            'divergence': (obv - obv_sma) / abs(obv_sma) * 100 if obv_sma != 0 else 0,
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='mfi_bands',
    feature_config=['mfi'],  # Simple: just declare we need MFI features
    param_feature_mapping={
        'mfi_period': 'mfi_{mfi_period}'
    }
)
def mfi_bands(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Money Flow Index overbought/oversold bands strategy.
    
    Returns sustained signal based on MFI extreme levels:
    - -1: MFI > overbought (mean reversion short)
    - 1: MFI < oversold (mean reversion long)
    - 0: MFI in neutral zone
    """
    mfi_period = params.get('mfi_period', 14)
    overbought = params.get('overbought', 80)
    oversold = params.get('oversold', 20)
    
    # Get features
    mfi = features.get(f'mfi_{mfi_period}')
    
    if mfi is None:
        return None
    
    # Determine signal based on bands
    if mfi > overbought:
        signal_value = -1  # Overbought (mean reversion)
    elif mfi < oversold:
        signal_value = 1   # Oversold (mean reversion)
    else:
        signal_value = 0   # Neutral zone
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'mfi_bands',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'mfi_period': mfi_period,                 # Parameters for sparse storage separation
            'overbought': overbought,
            'oversold': oversold,
            'mfi': mfi,                               # Values for analysis
            'price': bar.get('close', 0),
            'volume': bar.get('volume', 0)
        }
    }


@strategy(
    name='vwap_deviation',
    feature_config=['vwap']  # Simple: just declare we need VWAP features (no parameters needed)
)
def vwap_deviation(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    VWAP deviation strategy.
    
    Returns sustained signal based on price vs VWAP with deviation bands:
    - -1: Price > upper_band (mean reversion short)
    - 1: Price < lower_band (mean reversion long) 
    - 0: Price within bands
    """
    std_multiplier = params.get('std_multiplier', 2.0)
    
    # Get features
    vwap = features.get('vwap')
    price = bar.get('close', 0)
    
    if vwap is None:
        return None
    
    # Calculate VWAP bands using percentage-based approach
    # Since VWAP doesn't have standard deviation bands built-in, use percentage bands
    band_pct = params.get('band_pct', 0.02)  # 2% default bands
    vwap_upper = vwap * (1 + band_pct)
    vwap_lower = vwap * (1 - band_pct)
    
    # Determine signal based on deviation
    if price > vwap_upper:
        signal_value = -1  # Mean reversion short
    elif price < vwap_lower:
        signal_value = 1   # Mean reversion long
    else:
        signal_value = 0   # Within bands
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'vwap_deviation',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'std_multiplier': std_multiplier,         # Parameters for sparse storage separation
            'price': price,                           # Values for analysis
            'vwap': vwap,
            'upper_band': vwap_upper,
            'lower_band': vwap_lower,
            'deviation_pct': (price - vwap) / vwap * 100 if vwap != 0 else 0
        }
    }


@strategy(
    name='chaikin_money_flow',
    feature_config=['cmf'],  # Simple: just declare we need CMF features
    param_feature_mapping={
        'period': 'cmf_{period}'
    }
)
def chaikin_money_flow(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Chaikin Money Flow strategy.
    
    Returns sustained signal based on CMF levels:
    - 1: CMF > threshold (buying pressure)
    - -1: CMF < -threshold (selling pressure)
    - 0: CMF neutral
    """
    cmf_period = params.get('period', 20)
    threshold = params.get('threshold', 0.05)
    
    # Get features
    cmf = features.get(f'cmf_{cmf_period}')
    
    if cmf is None:
        return None
    
    # Determine signal
    if cmf > threshold:
        signal_value = 1   # Buying pressure
    elif cmf < -threshold:
        signal_value = -1  # Selling pressure
    else:
        signal_value = 0   # Neutral
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'chaikin_money_flow',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'period': cmf_period,                     # Parameters for sparse storage separation
            'threshold': threshold,
            'cmf': cmf,                               # Values for analysis
            'price': bar.get('close', 0),
            'volume': bar.get('volume', 0)
        }
    }


@strategy(
    name='accumulation_distribution',
    feature_config=['ad', 'ad_ema'],  # Need A/D and EMA of A/D
    param_feature_mapping={
        'ad_ema_period': 'ad_ema_{ad_ema_period}'
    }
)
def accumulation_distribution(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Accumulation/Distribution Line crossover strategy.
    
    Returns sustained signal based on A/D line vs its EMA:
    - 1: A/D > EMA (accumulation phase)
    - -1: A/D < EMA (distribution phase)
    - 0: Equal
    """
    ad_ema_period = params.get('ad_ema_period', 20)
    
    # Get features
    ad_line = features.get('ad')
    ad_ema = features.get(f'ad_ema_{ad_ema_period}')
    
    if ad_line is None or ad_ema is None:
        return None
    
    # Determine signal
    if ad_line > ad_ema:
        signal_value = 1   # Accumulation
    elif ad_line < ad_ema:
        signal_value = -1  # Distribution
    else:
        signal_value = 0
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'accumulation_distribution',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'ad_ema_period': ad_ema_period,           # Parameters for sparse storage separation
            'ad_line': ad_line,                       # Values for analysis
            'ad_ema': ad_ema,
            'divergence': ad_line - ad_ema,
            'price': bar.get('close', 0),
            'volume': bar.get('volume', 0)
        }
    }