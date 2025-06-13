"""
Moving average and crossover-based trading strategies.

This module implements event-driven versions of crossover trading rules
using binary signal values (-1, 0, 1) with sustained signals that persist
as long as conditions are met. Signal strength/sizing is handled by
separate components (risk module).
"""

from typing import Dict, Any, Optional
import logging
from ...core.components.discovery import strategy

logger = logging.getLogger(__name__)


@strategy(
    name='rule1_ma_crossover',
    feature_config={
        'sma': {
            'params': ['fast_period', 'slow_period'],
            'defaults': {'fast_period': 10, 'slow_period': 20}
        }
    }
)
def rule1_ma_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Rule 1: Simple Moving Average Crossover with sustained signals.
    
    Signal values:
    - 1: Fast MA > Slow MA (sustained)
    - -1: Fast MA < Slow MA (sustained)
    - 0: MAs equal (rare)
    
    Args:
        features: Calculated features from FeatureHub
        bar: Current bar data
        params: Strategy parameters (fast_period, slow_period)
        
    Returns:
        Signal dict only when signal value changes
    """
    fast_period = params.get('fast_period', 10)
    slow_period = params.get('slow_period', 20)
    
    # Get features
    fast_ma = features.get(f'sma_{fast_period}')
    slow_ma = features.get(f'sma_{slow_period}')
    
    if fast_ma is None or slow_ma is None:
        return None
    
    # Determine current signal state
    if fast_ma > slow_ma:
        signal_value = 1
    elif fast_ma < slow_ma:
        signal_value = -1
    else:
        signal_value = 0
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
            'symbol_timeframe': f"{symbol}_{timeframe}",
            'signal_value': signal_value,
            'timestamp': bar.get('timestamp'),
            'strategy_id': 'rule1_ma_crossover',
            'metadata': {
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'price': bar.get('close', 0),
                'separation_pct': abs(fast_ma - slow_ma) / slow_ma * 100 if slow_ma != 0 else 0
            }
    }


@strategy(
    name='rule2_ema_ma_crossover',
    feature_config={
        'ema': {
            'params': ['ema_period'],
            'defaults': {'ema_period': 10}
        },
        'sma': {
            'params': ['ma_period'],
            'defaults': {'ma_period': 20}
        }
    }
)
def rule2_ema_ma_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Rule 2: EMA vs MA Crossover with sustained signals.
    
    Signal values:
    - 1: EMA > SMA
    - -1: EMA < SMA
    - 0: Equal
    """
    ema_period = params.get('ema_period', 10)
    ma_period = params.get('ma_period', 20)
    
    # Get features
    ema = features.get(f'ema_{ema_period}')
    sma = features.get(f'sma_{ma_period}')
    
    if ema is None or sma is None:
        return None
    
    # Determine signal
    if ema > sma:
        signal_value = 1
    elif ema < sma:
        signal_value = -1
    else:
        signal_value = 0
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'rule2_ema_ma_crossover',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'ema': ema,
            'sma': sma,
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='rule3_ema_ema_crossover',
    feature_config={
        'ema': {
            'params': ['fast_ema_period', 'slow_ema_period'],
            'defaults': {'fast_ema_period': 10, 'slow_ema_period': 20}
        }
    }
)
def rule3_ema_ema_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Rule 3: EMA vs EMA Crossover with sustained signals.
    """
    fast_period = params.get('fast_ema_period', 10)
    slow_period = params.get('slow_ema_period', 20)
    
    # Get features
    fast_ema = features.get(f'ema_{fast_period}')
    slow_ema = features.get(f'ema_{slow_period}')
    
    if fast_ema is None or slow_ema is None:
        return None
    
    # Determine signal
    if fast_ema > slow_ema:
        signal_value = 1
    elif fast_ema < slow_ema:
        signal_value = -1
    else:
        signal_value = 0
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
            'symbol_timeframe': f"{symbol}_{timeframe}",
            'signal_value': signal_value,
            'timestamp': bar.get('timestamp'),
            'strategy_id': 'rule3_ema_ema_crossover',
            'metadata': {
                'fast_ema': fast_ema,
                'slow_ema': slow_ema,
                'price': bar.get('close', 0)
            }
    }


@strategy(
    name='rule4_dema_ma_crossover',
    feature_config={
        'dema': {
            'params': ['dema_period'],
            'defaults': {'dema_period': 10}
        },
        'sma': {
            'params': ['ma_period'],
            'defaults': {'ma_period': 20}
        }
    }
)
def rule4_dema_ma_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Rule 4: DEMA vs MA Crossover with sustained signals.
    """
    dema_period = params.get('dema_period', 10)
    ma_period = params.get('ma_period', 20)
    
    # Get features
    dema = features.get(f'dema_{dema_period}')
    sma = features.get(f'sma_{ma_period}')
    
    if dema is None or sma is None:
        return None
    
    # Determine signal
    if dema > sma:
        signal_value = 1
    elif dema < sma:
        signal_value = -1
    else:
        signal_value = 0
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
            'symbol_timeframe': f"{symbol}_{timeframe}",
            'signal_value': signal_value,
            'timestamp': bar.get('timestamp'),
            'strategy_id': 'rule4_dema_ma_crossover',
            'metadata': {
                'dema': dema,
                'sma': sma,
                'price': bar.get('close', 0)
            }
    }


@strategy(
    name='rule5_dema_dema_crossover',
    feature_config={
        'dema': {
            'params': ['fast_dema_period', 'slow_dema_period'],
            'defaults': {'fast_dema_period': 10, 'slow_dema_period': 20}
        }
    }
)
def rule5_dema_dema_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Rule 5: DEMA vs DEMA Crossover with sustained signals.
    """
    fast_period = params.get('fast_dema_period', 10)
    slow_period = params.get('slow_dema_period', 20)
    
    # Get features
    fast_dema = features.get(f'dema_{fast_period}')
    slow_dema = features.get(f'dema_{slow_period}')
    
    if fast_dema is None or slow_dema is None:
        return None
    
    # Determine signal
    if fast_dema > slow_dema:
        signal_value = 1
    elif fast_dema < slow_dema:
        signal_value = -1
    else:
        signal_value = 0
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
            'symbol_timeframe': f"{symbol}_{timeframe}",
            'signal_value': signal_value,
            'timestamp': bar.get('timestamp'),
            'strategy_id': 'rule5_dema_dema_crossover',
            'metadata': {
                'fast_dema': fast_dema,
                'slow_dema': slow_dema,
                'price': bar.get('close', 0)
            }
    }


@strategy(
    name='rule6_tema_ma_crossover',
    feature_config={
        'tema': {
            'params': ['tema_period'],
            'defaults': {'tema_period': 10}
        },
        'sma': {
            'params': ['ma_period'],
            'defaults': {'ma_period': 20}
        }
    }
)
def rule6_tema_ma_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Rule 6: TEMA vs MA Crossover with sustained signals.
    """
    tema_period = params.get('tema_period', 10)
    ma_period = params.get('ma_period', 20)
    
    # Get features
    tema = features.get(f'tema_{tema_period}')
    sma = features.get(f'sma_{ma_period}')
    
    if tema is None or sma is None:
        return None
    
    # Determine signal
    if tema > sma:
        signal_value = 1
    elif tema < sma:
        signal_value = -1
    else:
        signal_value = 0
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
            'symbol_timeframe': f"{symbol}_{timeframe}",
            'signal_value': signal_value,
            'timestamp': bar.get('timestamp'),
            'strategy_id': 'rule6_tema_ma_crossover',
            'metadata': {
                'tema': tema,
                'sma': sma,
                'price': bar.get('close', 0)
            }
    }


@strategy(
    name='rule7_stochastic_crossover',
    feature_config={
        'stochastic': {
            'params': ['k_period', 'd_period'],
            'defaults': {'k_period': 14, 'd_period': 3}
        }
    }
)
def rule7_stochastic_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Rule 7: Stochastic Oscillator Crossover with sustained signals.
    
    Signal based on %K vs %D relationship, not just crossovers.
    """
    k_period = params.get('k_period', 14)
    d_period = params.get('d_period', 3)
    
    # Get features
    stoch_k = features.get(f'stochastic_{k_period}_{d_period}_k')
    stoch_d = features.get(f'stochastic_{k_period}_{d_period}_d')
    
    if stoch_k is None or stoch_d is None:
        return None
    
    # Determine signal
    if stoch_k > stoch_d:
        signal_value = 1
    elif stoch_k < stoch_d:
        signal_value = -1
    else:
        signal_value = 0
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
            'symbol_timeframe': f"{symbol}_{timeframe}",
            'signal_value': signal_value,
            'timestamp': bar.get('timestamp'),
            'strategy_id': 'rule7_stochastic_crossover',
            'metadata': {
                'stoch_k': stoch_k,
                'stoch_d': stoch_d,
                'price': bar.get('close', 0)
            }
    }


@strategy(
    name='rule8_vortex_crossover',
    feature_config={
        'vortex': {
            'params': ['vortex_period'],
            'defaults': {'vortex_period': 14}
        }
    }
)
def rule8_vortex_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Rule 8: Vortex Indicator Crossover with sustained signals.
    """
    vortex_period = params.get('vortex_period', 14)
    
    # Get features
    vi_plus = features.get(f'vortex_{vortex_period}_vi_plus')
    vi_minus = features.get(f'vortex_{vortex_period}_vi_minus')
    
    if vi_plus is None or vi_minus is None:
        return None
    
    # Determine signal
    if vi_plus > vi_minus:
        signal_value = 1
    elif vi_plus < vi_minus:
        signal_value = -1
    else:
        signal_value = 0
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
            'symbol_timeframe': f"{symbol}_{timeframe}",
            'signal_value': signal_value,
            'timestamp': bar.get('timestamp'),
            'strategy_id': 'rule8_vortex_crossover',
            'metadata': {
                'vi_plus': vi_plus,
                'vi_minus': vi_minus,
                'price': bar.get('close', 0)
            }
    }


@strategy(
    name='rule9_ichimoku_crossover',
    feature_config={
        'ichimoku': {
            'params': ['conversion_period', 'base_period'],
            'defaults': {'conversion_period': 9, 'base_period': 26}
        }
    }
)
def rule9_ichimoku_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Rule 9: Ichimoku Cloud Crossover with sustained signals.
    
    Simplified: Signal based on price position relative to cloud.
    """
    conversion_period = params.get('conversion_period', 9)
    base_period = params.get('base_period', 26)
    
    # Get Ichimoku components
    span_a = features.get(f'ichimoku_{conversion_period}_{base_period}_52_senkou_span_a')
    span_b = features.get(f'ichimoku_{conversion_period}_{base_period}_52_senkou_span_b')
    price = bar.get('close', 0)
    
    if span_a is None or span_b is None:
        return None
    
    # Determine cloud boundaries
    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)
    
    # Determine signal
    if price > cloud_top:
        signal_value = 1  # Price above cloud
    elif price < cloud_bottom:
        signal_value = -1  # Price below cloud
    else:
        signal_value = 0  # Price in cloud (neutral)
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
            'symbol_timeframe': f"{symbol}_{timeframe}",
            'signal_value': signal_value,
            'timestamp': bar.get('timestamp'),
            'strategy_id': 'rule9_ichimoku_crossover',
            'metadata': {
                'price': price,
                'cloud_top': cloud_top,
                'cloud_bottom': cloud_bottom,
                'senkou_span_a': span_a,
                'senkou_span_b': span_b
            }
    }