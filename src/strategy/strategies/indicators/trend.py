"""
Trend-based indicator strategies.

All trend strategies that generate signals based on trend strength,
direction, and changes.
"""

from typing import Dict, Any, Optional
import logging
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec

logger = logging.getLogger(__name__)


@strategy(
    name='adx_trend_strength',
    feature_discovery=lambda params: [
        FeatureSpec('adx', {'period': params.get('adx_period', 14)}, 'adx'),
        FeatureSpec('adx', {'period': params.get('adx_period', 14)}, 'di_plus'),
        FeatureSpec('adx', {'period': params.get('adx_period', 14)}, 'di_minus')
    ],
    parameter_space={
        'adx_period': {'type': 'int', 'range': (10, 50), 'default': 14},
        'adx_threshold': {'type': 'float', 'range': (15, 40), 'default': 25},
        'di_period': {'type': 'int', 'range': (10, 50), 'default': 14}
    }
)
def adx_trend_strength(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    ADX trend strength with DI crossover strategy.
    
    Returns sustained signal based on ADX strength and DI direction:
    - 1: Strong trend (ADX > threshold) with DI+ > DI- (bullish)
    - -1: Strong trend (ADX > threshold) with DI+ < DI- (bearish)
    - 0: Weak trend (ADX < threshold) or no clear direction
    """
    adx_period = params.get('adx_period', 14)
    di_period = params.get('di_period', 14)
    adx_threshold = params.get('adx_threshold', 25)
    
    # Get features
    adx = features.get(f'adx_{adx_period}_adx')
    di_plus = features.get(f'adx_{adx_period}_di_plus')
    di_minus = features.get(f'adx_{adx_period}_di_minus')
    
    if adx is None or di_plus is None or di_minus is None:
        logger.debug(f"adx_trend_strength waiting for features: adx={adx is not None}, di_plus={di_plus is not None}, di_minus={di_minus is not None}")
        return None
    
    # Determine signal
    if adx > adx_threshold:
        # Strong trend
        if di_plus > di_minus:
            signal_value = 1   # Bullish trend
        else:
            signal_value = -1  # Bearish trend
    else:
        signal_value = 0  # Weak or no trend
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'adx_trend_strength',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'adx_period': adx_period,                  # Parameters for sparse storage separation
            'di_period': di_period,
            'adx_threshold': adx_threshold,
            'adx': adx,                                # Values for analysis
            'di_plus': di_plus,
            'di_minus': di_minus,
            'trend_strength': 'strong' if adx > adx_threshold else 'weak',
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='parabolic_sar',
    feature_discovery=lambda params: [
        FeatureSpec('psar', {
            'af_start': params.get('af_start', 0.02),
            'af_max': params.get('af_max', 0.2)
        })
    ],
    parameter_space={
        'af_max': {'type': 'float', 'range': (0.1, 0.5), 'default': 0.2},
        'af_start': {'type': 'float', 'range': (0.01, 0.1), 'default': 0.02}
    }
)
def parabolic_sar(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parabolic SAR trend following strategy.
    
    Returns sustained signal based on price vs PSAR:
    - 1: Price > PSAR (uptrend)
    - -1: Price < PSAR (downtrend)
    - 0: Price = PSAR (rare)
    """
    psar_af = params.get('af_start', 0.02)
    psar_max_af = params.get('af_max', 0.2)
    
    # Get features
    psar = features.get(f'psar_{psar_af}_{psar_max_af}')
    price = bar.get('close', 0)
    
    if psar is None:
        return None
    
    # Determine signal
    if price > psar:
        signal_value = 1   # Uptrend
    elif price < psar:
        signal_value = -1  # Downtrend
    else:
        signal_value = 0
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'parabolic_sar',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'af_start': psar_af,                       # Parameters for sparse storage separation
            'af_max': psar_max_af,
            'price': price,                            # Values for analysis
            'psar': psar,
            'distance': abs(price - psar),
            'distance_pct': abs(price - psar) / price * 100 if price != 0 else 0
        }
    }


@strategy(
    name='aroon_crossover',
    feature_discovery=lambda params: [
        FeatureSpec('aroon', {'period': params.get('period', 25)}, 'up'),
        FeatureSpec('aroon', {'period': params.get('period', 25)}, 'down')
    ],
    parameter_space={
        'period': {'type': 'int', 'range': (10, 50), 'default': 25}
    }
)
def aroon_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Aroon indicator crossover strategy.
    
    Returns sustained signal based on Aroon Up vs Aroon Down:
    - 1: Aroon Up > Aroon Down (uptrend)
    - -1: Aroon Up < Aroon Down (downtrend)
    - 0: Equal (no clear trend)
    """
    aroon_period = params.get('period', 25)
    
    # Get features
    aroon_up = features.get(f'aroon_{aroon_period}_up')
    aroon_down = features.get(f'aroon_{aroon_period}_down')
    
    if aroon_up is None or aroon_down is None:
        return None
    
    # Determine signal
    if aroon_up > aroon_down:
        signal_value = 1   # Uptrend
    elif aroon_up < aroon_down:
        signal_value = -1  # Downtrend
    else:
        signal_value = 0   # No clear trend
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'aroon_crossover',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'period': aroon_period,                    # Parameters for sparse storage separation
            'aroon_up': aroon_up,                      # Values for analysis
            'aroon_down': aroon_down,
            'aroon_oscillator': aroon_up - aroon_down,
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='supertrend',
    feature_discovery=lambda params: [
        FeatureSpec('supertrend', {
            'period': params.get('period', 10),
            'multiplier': params.get('multiplier', 3.0)
        })
    ],
    parameter_space={
        'multiplier': {'type': 'float', 'range': (1.0, 5.0), 'default': 3.0},
        'period': {'type': 'int', 'range': (5, 30), 'default': 10}
    }
)
def supertrend(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Supertrend indicator strategy.
    
    Returns sustained signal based on price vs Supertrend:
    - 1: Price > Supertrend (uptrend)
    - -1: Price < Supertrend (downtrend)
    - 0: Price = Supertrend (rare)
    """
    period = params.get('period', 10)
    multiplier = params.get('multiplier', 3.0)
    
    # Get features
    # SuperTrend returns multiple values, need to access the sub-keys
    supertrend_value = features.get(f'supertrend_{period}_{multiplier}_supertrend')
    supertrend_direction = features.get(f'supertrend_{period}_{multiplier}_trend')
    price = bar.get('close', 0)
    
    if supertrend_value is None:
        return None
    
    # Use direction if available, otherwise compare price to supertrend
    if supertrend_direction is not None:
        signal_value = 1 if supertrend_direction == 1 else -1  # trend: 1 for up, -1 for down
    else:
        if price > supertrend_value:
            signal_value = 1   # Uptrend
        elif price < supertrend_value:
            signal_value = -1  # Downtrend
        else:
            signal_value = 0
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'supertrend',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'period': period,                          # Parameters for sparse storage separation
            'multiplier': multiplier,
            'price': price,                            # Values for analysis
            'supertrend': supertrend_value,
            'direction': supertrend_direction if supertrend_direction is not None else signal_value,
            'distance': abs(price - supertrend_value) if supertrend_value is not None else 0
        }
    }


@strategy(
    name='linear_regression_slope',
    feature_discovery=lambda params: [
        FeatureSpec('linear_regression', {'period': params.get('period', 20)}, 'slope')
    ],
    parameter_space={
        'period': {'type': 'int', 'range': (10, 50), 'default': 20},
        'threshold': {'type': 'float', 'range': (-0.5, 0.5), 'default': 0.0}
    }
)
def linear_regression_slope(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Linear regression slope strategy.
    
    Returns sustained signal based on regression slope:
    - 1: Positive slope > threshold (uptrend)
    - -1: Negative slope < -threshold (downtrend)
    - 0: Flat (abs(slope) < threshold)
    """
    lr_period = params.get('period', 20)
    slope_threshold = params.get('threshold', 0.0)
    
    # Get features
    lr_slope = features.get(f'linear_regression_{lr_period}_slope')
    lr_intercept = features.get(f'linear_regression_{lr_period}_intercept')
    lr_r2 = features.get(f'linear_regression_{lr_period}_r2')
    
    if lr_slope is None:
        return None
    
    # Determine signal based on slope
    if lr_slope > slope_threshold:
        signal_value = 1   # Uptrend
    elif lr_slope < -slope_threshold:
        signal_value = -1  # Downtrend
    else:
        signal_value = 0   # Flat/ranging
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'linear_regression_slope',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'period': lr_period,                       # Parameters for sparse storage separation
            'threshold': slope_threshold,
            'slope': lr_slope,                         # Values for analysis
            'intercept': lr_intercept if lr_intercept is not None else 0,
            'r_squared': lr_r2 if lr_r2 is not None else 0,
            'price': bar.get('close', 0)
        }
    }