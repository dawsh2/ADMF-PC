"""
Session VWAP mean reversion strategy.

Uses SessionVWAP that resets at market open for intraday mean reversion.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='session_vwap_deviation',
    feature_discovery=lambda params: [
        FeatureSpec('session_vwap', {
            'session_start_hour': params.get('session_start_hour', 9),
            'session_start_minute': params.get('session_start_minute', 30),
            'reset_on_gap_minutes': params.get('reset_on_gap_minutes', 60)
        }),
        FeatureSpec('atr', {'period': params.get('atr_period', 14)})
        if params.get('use_atr_bands', False) else None
    ],
    parameter_space={
        'band_pct': {
            'type': 'float', 
            'range': (0.001, 0.02), 
            'default': 0.005,
            'granularity': 5,  # Test 0.1%, 0.5%, 1%, 1.5%, 2%
            'description': 'Percentage band width from VWAP'
        },
        'use_atr_bands': {
            'type': 'bool',
            'default': False,
            'description': 'Use ATR-based bands instead of percentage'
        },
        'atr_multiplier': {
            'type': 'float',
            'range': (0.5, 3.0),
            'default': 1.0,
            'granularity': 4,  # Test 0.5, 1.0, 2.0, 3.0
            'description': 'ATR multiplier for adaptive bands'
        },
        'atr_period': {
            'type': 'int',
            'range': (7, 21),
            'default': 14,
            'granularity': 3,
            'description': 'ATR calculation period'
        },
        'session_start_hour': {
            'type': 'int',
            'range': (9, 9),
            'default': 9,
            'description': 'Market open hour (24h format)'
        },
        'session_start_minute': {
            'type': 'int', 
            'range': (30, 30),
            'default': 30,
            'description': 'Market open minute'
        },
        'reset_on_gap_minutes': {
            'type': 'int',
            'range': (60, 120),
            'default': 60,
            'description': 'Reset VWAP if time gap exceeds this'
        }
    },
    strategy_type='mean_reversion',
    tags=['mean_reversion', 'vwap', 'session', 'intraday']
)
def session_vwap_deviation(features: Dict[str, Any], bar: Dict[str, Any], 
                          params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Session VWAP deviation strategy for intraday mean reversion.
    
    Returns sustained signal based on price vs session VWAP with deviation bands:
    - -1: Price > upper_band (mean reversion short)
    - 1: Price < lower_band (mean reversion long) 
    - 0: Price within bands
    
    The session VWAP resets at market open (9:30 AM by default), making it
    ideal for intraday mean reversion as it tracks the day's volume-weighted price.
    """
    use_atr_bands = params.get('use_atr_bands', False)
    
    # Get features
    session_vwap = features.get('session_vwap')
    price = bar.get('close', 0)
    
    if session_vwap is None:
        return None
    
    # Calculate bands based on method
    atr = None  # Initialize for metadata
    if use_atr_bands:
        # ATR-based bands (adaptive to volatility)
        atr_period = params.get('atr_period', 14)
        atr = features.get(f'atr_{atr_period}')
        
        if atr is None:
            return None
            
        atr_multiplier = params.get('atr_multiplier', 1.0)
        band_width = atr * atr_multiplier
        vwap_upper = session_vwap + band_width
        vwap_lower = session_vwap - band_width
        band_parameter = atr_multiplier
    else:
        # Percentage-based bands (fixed percentage)
        band_pct = params.get('band_pct', 0.005)  # 0.5% default bands
        vwap_upper = session_vwap * (1 + band_pct)
        vwap_lower = session_vwap * (1 - band_pct)
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
        'strategy_id': 'session_vwap_deviation',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'band_method': 'atr' if use_atr_bands else 'percentage',
            'band_parameter': band_parameter,
            'price': price,
            'session_vwap': session_vwap,
            'upper_band': vwap_upper,
            'lower_band': vwap_lower,
            'deviation_pct': (price - session_vwap) / session_vwap * 100 if session_vwap != 0 else 0,
            'atr': atr if use_atr_bands else None,
            'session_params': {
                'start_hour': params.get('session_start_hour', 9),
                'start_minute': params.get('session_start_minute', 30),
                'reset_gap': params.get('reset_on_gap_minutes', 60)
            }
        }
    }