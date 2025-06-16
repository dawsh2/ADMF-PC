"""
Trend-based indicator strategies.

All trend strategies that generate signals based on trend strength,
direction, and changes.
"""

from typing import Dict, Any, Optional
import logging
from ....core.components.discovery import strategy

logger = logging.getLogger(__name__)


@strategy(
    name='adx_trend_strength',
    feature_config=['adx'],  # Simple: just declare we need ADX features (includes DI)
    param_feature_mapping={
        'adx_period': 'adx_{adx_period}',
        'di_period': 'adx_{di_period}'
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
    feature_config=['psar'],  # Simple: just declare we need Parabolic SAR features
    param_feature_mapping={
        'af_start': 'psar_{af_start}_{af_max}',
        'af_max': 'psar_{af_start}_{af_max}'
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
    feature_config=['aroon'],  # Simple: just declare we need Aroon features
    param_feature_mapping={
        'period': 'aroon_{period}'
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
    feature_config=['supertrend'],  # Simple: just declare we need Supertrend features
    param_feature_mapping={
        'period': 'supertrend_{period}_{multiplier}',
        'multiplier': 'supertrend_{period}_{multiplier}'
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
    feature_config=['linear_regression'],  # Simple: just declare we need Linear Regression features
    param_feature_mapping={
        'period': 'linear_regression_{period}'
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
    slope_threshold = params.get('slope_threshold', 0.0)
    
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
            'slope_threshold': slope_threshold,
            'slope': lr_slope,                         # Values for analysis
            'intercept': lr_intercept if lr_intercept is not None else 0,
            'r_squared': lr_r2 if lr_r2 is not None else 0,
            'price': bar.get('close', 0)
        }
    }