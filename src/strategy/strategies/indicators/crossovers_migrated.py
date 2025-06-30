"""
Migrated crossover strategies using FeatureSpec.

This demonstrates the new feature system with:
1. Explicit feature requirements via FeatureSpec
2. Dynamic feature discovery based on parameters
3. ValidatedFeatures container with guaranteed feature existence
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec, ValidatedFeatures


# Example 1: Dynamic discovery approach
@strategy(
    name='sma_crossover_v2',
    feature_discovery=lambda params: [
        FeatureSpec('sma', {'period': params.get('fast_period', 10)}),
        FeatureSpec('sma', {'period': params.get('slow_period', 20)})
    ]
)
def sma_crossover_v2(features: ValidatedFeatures, bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    SMA crossover strategy using new FeatureSpec system.
    
    Features are dynamically discovered based on parameters and validated.
    No more silent failures - features are guaranteed to exist.
    """
    fast_period = params.get('fast_period', 10)
    slow_period = params.get('slow_period', 20)
    
    # Features are guaranteed to exist - no None checks needed!
    fast_sma = features[f'sma_{fast_period}']
    slow_sma = features[f'sma_{slow_period}']
    
    # Determine signal state
    if fast_sma > slow_sma:
        signal_value = 1
    elif fast_sma < slow_sma:
        signal_value = -1
    else:
        signal_value = 0
    
    # Always return current signal state
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'sma_crossover_v2',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'fast_sma': fast_sma,
            'slow_sma': slow_sma,
            'price': bar.get('close', 0),
            'separation_pct': abs(fast_sma - slow_sma) / slow_sma * 100 if slow_sma != 0 else 0
        }
    }


# Example 2: Static requirements with defaults
@strategy(
    name='ema_sma_crossover_v2',
    required_features=[
        FeatureSpec('ema', {'period': 10}),  # Static default
        FeatureSpec('sma', {'period': 20})   # Static default
    ]
)
def ema_sma_crossover_static(features: ValidatedFeatures, bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    EMA vs SMA crossover with static feature requirements.
    
    This version always uses EMA(10) and SMA(20) regardless of parameters.
    Good for strategies with fixed indicators.
    """
    # Features are guaranteed to exist
    ema = features['ema_10']
    sma = features['sma_20']
    
    # Determine signal
    if ema > sma:
        signal_value = 1
    elif ema < sma:
        signal_value = -1
    else:
        signal_value = 0
    
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'ema_sma_crossover_static',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'ema': ema,
            'sma': sma,
            'price': bar.get('close', 0),
            'separation_pct': abs(ema - sma) / sma * 100 if sma != 0 else 0
        }
    }


# Example 3: Multi-output feature (MACD)
@strategy(
    name='macd_crossover_v2',
    feature_discovery=lambda params: [
        FeatureSpec('macd', {
            'fast_period': params.get('fast_period', 12),
            'slow_period': params.get('slow_period', 26),
            'signal_period': params.get('signal_period', 9)
        }, 'macd'),  # Specify we want the 'macd' component
        FeatureSpec('macd', {
            'fast_period': params.get('fast_period', 12),
            'slow_period': params.get('slow_period', 26),
            'signal_period': params.get('signal_period', 9)
        }, 'signal')  # Specify we want the 'signal' component
    ]
)
def macd_crossover_v2(features: ValidatedFeatures, bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    MACD crossover demonstrating multi-output features.
    
    Shows how to request specific components of multi-output indicators.
    """
    fast = params.get('fast_period', 12)
    slow = params.get('slow_period', 26)
    signal = params.get('signal_period', 9)
    
    # Exact, deterministic feature names
    macd_line = features[f'macd_{fast}_{slow}_{signal}_macd']
    signal_line = features[f'macd_{fast}_{slow}_{signal}_signal']
    
    # Generate signal based on MACD vs Signal crossover
    if macd_line > signal_line:
        signal_value = 1
    elif macd_line < signal_line:
        signal_value = -1
    else:
        signal_value = 0
    
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'macd_crossover_v2',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'fast_period': fast,
            'slow_period': slow,
            'signal_period': signal,
            'macd': macd_line,
            'signal': signal_line,
            'histogram': macd_line - signal_line,
            'price': bar.get('close', 0)
        }
    }


# Example 4: Hybrid approach - some static, some dynamic
@strategy(
    name='adaptive_crossover',
    required_features=[
        FeatureSpec('atr', {'period': 14})  # Always need ATR(14) for volatility
    ],
    feature_discovery=lambda params: [
        FeatureSpec('ema', {'period': params.get('adaptive_period', 20)}),
        FeatureSpec('sma', {'period': params.get('adaptive_period', 20) * 2})  # Double the period
    ]
)
def adaptive_crossover(features: ValidatedFeatures, bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Adaptive crossover showing hybrid static + dynamic features.
    
    Always uses ATR(14) but adapts MA periods based on parameters.
    """
    period = params.get('adaptive_period', 20)
    
    # Static feature
    atr = features['atr_14']
    
    # Dynamic features
    ema = features[f'ema_{period}']
    sma = features[f'sma_{period * 2}']
    
    # Adjust signal strength based on volatility
    base_signal = 1 if ema > sma else -1 if ema < sma else 0
    
    # Scale by volatility
    current_price = bar.get('close', 0)
    volatility_factor = (atr / current_price) * 100 if current_price > 0 else 0
    
    # Reduce signal in high volatility
    if volatility_factor > 2.0:  # High volatility
        signal_strength = 0.5
    else:
        signal_strength = 1.0
    
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    
    return {
        'signal_value': base_signal * signal_strength,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'adaptive_crossover',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'adaptive_period': period,
            'ema': ema,
            'sma': sma,
            'atr': atr,
            'volatility_factor': volatility_factor,
            'signal_strength': signal_strength,
            'price': bar.get('close', 0)
        }
    }