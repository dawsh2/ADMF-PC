"""
Volume-based indicator strategies.

All volume strategies that generate signals based on volume analysis
and volume-price relationships.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='obv_trend',
    feature_discovery=lambda params: [
        FeatureSpec('obv', {})
    ],
    parameter_space={
        'obv_threshold': {'type': 'float', 'range': (0, 1000000), 'default': 0}
    }
)
def obv_trend(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    On-Balance Volume trend strategy.
    
    For now, returns signal based on OBV direction:
    - 1: OBV > threshold (accumulation)
    - -1: OBV < -threshold (distribution)
    - 0: Between thresholds
    
    TODO: Add SMA comparison when composite features are supported
    """
    obv_threshold = params.get('obv_threshold', 0)
    
    # Get features
    obv = features.get('obv')
    
    if obv is None:
        return None
    
    # Determine signal based on OBV value
    if obv > obv_threshold:
        signal_value = 1  # Accumulation
    elif obv < -obv_threshold:
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
            'obv_threshold': obv_threshold,         # Parameters for sparse storage separation
            'obv': obv,                               # Values for analysis
            'signal': 'accumulation' if signal_value == 1 else 'distribution' if signal_value == -1 else 'neutral',
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='mfi_bands',
    feature_discovery=lambda params: [
        FeatureSpec('mfi', {'period': params.get('mfi_period', 14)})
    ],
    parameter_space={
        'mfi_period': {'type': 'int', 'range': (7, 30), 'default': 14},
        'overbought': {'type': 'float', 'range': (60, 90), 'default': 80},
        'oversold': {'type': 'float', 'range': (10, 40), 'default': 20}
    },
    strategy_type='mean_reversion'
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
    feature_discovery=lambda params: [
        FeatureSpec('vwap', {})
    ] + ([FeatureSpec('atr', {'period': params.get('atr_period', 14)})] if params.get('use_atr_bands', False) else []),
    parameter_space={
        'band_pct': {'type': 'float', 'range': (0.002, 0.02), 'default': 0.005},  # 0.2% to 2%, default 0.5%
        'atr_multiplier': {'type': 'float', 'range': (0.5, 2.0), 'default': 1.0},  # ATR multiplier for bands
        'use_atr_bands': {'type': 'bool', 'default': False},  # Use ATR-based bands instead of percentage
        'atr_period': {'type': 'int', 'range': (10, 30), 'default': 14}
    },
    strategy_type='mean_reversion',
    tags=['mean_reversion', 'vwap', 'intraday']
)
def vwap_deviation(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    VWAP deviation strategy with percentage or ATR-based bands.
    
    Returns sustained signal based on price vs VWAP with deviation bands:
    - -1: Price > upper_band (mean reversion short)
    - 1: Price < lower_band (mean reversion long) 
    - 0: Price within bands
    
    Supports two band calculation methods:
    1. Percentage-based: Fixed percentage from VWAP
    2. ATR-based: Adaptive bands based on volatility
    """
    use_atr_bands = params.get('use_atr_bands', False)
    
    # Get features
    vwap = features.get('vwap')
    price = bar.get('close', 0)
    
    if vwap is None:
        return None
    
    # Calculate bands based on method
    atr = None  # Initialize for metadata
    if use_atr_bands:
        # ATR-based bands (adaptive to volatility)
        atr_period = params.get('atr_period', 14)
        atr = features.get(f'atr_{atr_period}') or features.get('atr')
        
        if atr is None:
            return None
            
        atr_multiplier = params.get('atr_multiplier', 1.0)
        band_width = atr * atr_multiplier
        vwap_upper = vwap + band_width
        vwap_lower = vwap - band_width
        band_parameter = atr_multiplier
    else:
        # Percentage-based bands (fixed percentage)
        band_pct = params.get('band_pct', 0.005)  # 0.5% default bands
        vwap_upper = vwap * (1 + band_pct)
        vwap_lower = vwap * (1 - band_pct)
        band_parameter = band_pct
    
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
            'band_method': 'atr' if use_atr_bands else 'percentage',  # Band calculation method
            'band_parameter': atr_multiplier if use_atr_bands else band_pct,  # Relevant parameter
            'price': price,                           # Values for analysis
            'vwap': vwap,
            'upper_band': vwap_upper,
            'lower_band': vwap_lower,
            'deviation_pct': (price - vwap) / vwap * 100 if vwap != 0 else 0,
            'atr': atr if use_atr_bands else None    # Include ATR if used
        }
    }


@strategy(
    name='chaikin_money_flow',
    feature_discovery=lambda params: [
        FeatureSpec('cmf', {'period': params.get('period', 20)})
    ],
    parameter_space={
        'period': {'type': 'int', 'range': (10, 50), 'default': 20},
        'threshold': {'type': 'float', 'range': (0.0, 0.2), 'default': 0.05}
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
    feature_discovery=lambda params: [
        FeatureSpec('ad', {})
    ],
    parameter_space={
        'ema_period': {'type': 'int', 'range': (10, 50), 'default': 20}
    }
)
def accumulation_distribution(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Accumulation/Distribution Line trend strategy.
    
    For now, returns signal based on A/D line direction:
    - 1: A/D > 0 (accumulation phase)
    - -1: A/D < 0 (distribution phase)
    - 0: A/D = 0
    
    TODO: Add EMA comparison when composite features are supported
    """
    # Get features
    ad_line = features.get('ad')
    
    if ad_line is None:
        return None
    
    # Determine signal based on A/D line value
    if ad_line > 0:
        signal_value = 1   # Accumulation
    elif ad_line < 0:
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
            'ad_line': ad_line,                       # Values for analysis
            'signal': 'accumulation' if signal_value == 1 else 'distribution' if signal_value == -1 else 'neutral',
            'price': bar.get('close', 0),
            'volume': bar.get('volume', 0)
        }
    }