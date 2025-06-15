"""
Crossover-based indicator strategies.

All crossover strategies that generate signals based on one indicator 
crossing above/below another indicator or reference line.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy


@strategy(
    name='sma_crossover',
    feature_config=['sma'],  # Simple: just declare we need SMA features
    param_feature_mapping={  # Map parameters to specific feature names
        'fast_period': 'sma_{fast_period}',
        'slow_period': 'sma_{slow_period}'
    }
)
def sma_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    SMA crossover strategy - fast SMA vs slow SMA.
    
    Returns sustained signal based on SMA relationship:
    - 1: Fast SMA > Slow SMA (bullish)
    - -1: Fast SMA < Slow SMA (bearish)  
    - 0: Equal
    """
    fast_period = params.get('fast_period', 10)
    slow_period = params.get('slow_period', 20)
    
    # Get features
    fast_sma = features.get(f'sma_{fast_period}')
    slow_sma = features.get(f'sma_{slow_period}')
    
    if fast_sma is None or slow_sma is None:
        return None
    
    # Determine signal state
    if fast_sma > slow_sma:
        signal_value = 1
    elif fast_sma < slow_sma:
        signal_value = -1
    else:
        signal_value = 0
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')  # Default to 1m to match config
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'sma_crossover',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'fast_period': fast_period,     # Parameters for sparse storage separation
            'slow_period': slow_period,
            'fast_sma': fast_sma,          # Values for analysis
            'slow_sma': slow_sma,
            'price': bar.get('close', 0),
            'separation_pct': abs(fast_sma - slow_sma) / slow_sma * 100 if slow_sma != 0 else 0
        }
    }


@strategy(
    name='ema_sma_crossover',
    feature_config=['ema', 'sma'],  # Simple: declare we need EMA and SMA features
    param_feature_mapping={
        'ema_period': 'ema_{ema_period}',
        'sma_period': 'sma_{sma_period}'
    }
)
def ema_sma_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    EMA vs SMA crossover strategy.
    
    Returns sustained signal based on EMA vs SMA relationship:
    - 1: EMA > SMA
    - -1: EMA < SMA
    - 0: Equal
    """
    ema_period = params.get('ema_period', 10)
    sma_period = params.get('sma_period', 20)
    
    # Get features
    ema = features.get(f'ema_{ema_period}')
    sma = features.get(f'sma_{sma_period}')
    
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
    timeframe = bar.get('timeframe', '1m')  # Default to 1m to match config
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'ema_sma_crossover',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'ema_period': ema_period,      # Parameters for sparse storage separation
            'sma_period': sma_period,
            'ema': ema,                    # Values for analysis
            'sma': sma,
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='ema_crossover',
    feature_config=['ema'],  # Simple: just declare we need EMA features
    param_feature_mapping={
        'fast_ema_period': 'ema_{fast_ema_period}',
        'slow_ema_period': 'ema_{slow_ema_period}'
    }
)
def ema_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    EMA crossover strategy - fast EMA vs slow EMA.
    
    Returns sustained signal based on EMA relationship:
    - 1: Fast EMA > Slow EMA  
    - -1: Fast EMA < Slow EMA
    - 0: Equal
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
    timeframe = bar.get('timeframe', '1m')  # Default to 1m to match config
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'ema_crossover',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'fast_ema_period': fast_period,    # Parameters for sparse storage separation
            'slow_ema_period': slow_period,
            'fast_ema': fast_ema,              # Values for analysis
            'slow_ema': slow_ema,
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='dema_sma_crossover',
    feature_config=['dema', 'sma'],  # Simple: declare we need DEMA and SMA features
    param_feature_mapping={
        'dema_period': 'dema_{dema_period}',
        'sma_period': 'sma_{sma_period}'
    }
)
def dema_sma_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    DEMA vs SMA crossover strategy.
    
    Returns sustained signal based on DEMA vs SMA relationship:
    - 1: DEMA > SMA
    - -1: DEMA < SMA
    - 0: Equal
    """
    dema_period = params.get('dema_period', 10)
    sma_period = params.get('sma_period', 20)
    
    # Get features
    dema = features.get(f'dema_{dema_period}')
    sma = features.get(f'sma_{sma_period}')
    
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
    timeframe = bar.get('timeframe', '1m')  # Default to 1m to match config
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'dema_sma_crossover',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'dema_period': dema_period,        # Parameters for sparse storage separation
            'sma_period': sma_period,
            'dema': dema,                      # Values for analysis
            'sma': sma,
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='dema_crossover',
    feature_config=['dema'],  # Simple: just declare we need DEMA features
    param_feature_mapping={
        'fast_dema_period': 'dema_{fast_dema_period}',
        'slow_dema_period': 'dema_{slow_dema_period}'
    }
)
def dema_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    DEMA crossover strategy - fast DEMA vs slow DEMA.
    
    Returns sustained signal based on DEMA relationship:
    - 1: Fast DEMA > Slow DEMA
    - -1: Fast DEMA < Slow DEMA
    - 0: Equal
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
    timeframe = bar.get('timeframe', '1m')  # Default to 1m to match config
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'dema_crossover',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'fast_dema_period': fast_period,   # Parameters for sparse storage separation
            'slow_dema_period': slow_period,
            'fast_dema': fast_dema,            # Values for analysis
            'slow_dema': slow_dema,
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='tema_sma_crossover',
    feature_config=['tema', 'sma'],  # Simple: declare we need TEMA and SMA features
    param_feature_mapping={
        'tema_period': 'tema_{tema_period}',
        'sma_period': 'sma_{sma_period}'
    }
)
def tema_sma_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    TEMA vs SMA crossover strategy.
    
    Returns sustained signal based on TEMA vs SMA relationship:
    - 1: TEMA > SMA
    - -1: TEMA < SMA
    - 0: Equal
    """
    tema_period = params.get('tema_period', 10)
    sma_period = params.get('sma_period', 20)
    
    # Get features
    tema = features.get(f'tema_{tema_period}')
    sma = features.get(f'sma_{sma_period}')
    
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
    timeframe = bar.get('timeframe', '1m')  # Default to 1m to match config
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'tema_sma_crossover',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'tema_period': tema_period,        # Parameters for sparse storage separation
            'sma_period': sma_period,
            'tema': tema,                      # Values for analysis
            'sma': sma,
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='stochastic_crossover',
    feature_config=['stochastic'],  # Simple: just declare we need stochastic features
    param_feature_mapping={
        'k_period': 'stochastic_{k_period}_{d_period}',
        'd_period': 'stochastic_{k_period}_{d_period}'
    }
)
def stochastic_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Stochastic %K vs %D crossover strategy.
    
    Returns sustained signal based on Stochastic crossover:
    - 1: %K > %D
    - -1: %K < %D
    - 0: Equal
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
    timeframe = bar.get('timeframe', '1m')  # Default to 1m to match config
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'stochastic_crossover',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'k_period': k_period,              # Parameters for sparse storage separation
            'd_period': d_period,
            'stoch_k': stoch_k,                # Values for analysis
            'stoch_d': stoch_d,
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='vortex_crossover',
    feature_config=['vortex'],  # Simple: just declare we need vortex features
    param_feature_mapping={
        'vortex_period': 'vortex_{vortex_period}'
    }
)
def vortex_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Vortex VI+ vs VI- crossover strategy.
    
    Returns sustained signal based on Vortex crossover:
    - 1: VI+ > VI-
    - -1: VI+ < VI-
    - 0: Equal
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
    timeframe = bar.get('timeframe', '1m')  # Default to 1m to match config
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'vortex_crossover',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'vortex_period': vortex_period,    # Parameters for sparse storage separation
            'vi_plus': vi_plus,                # Values for analysis
            'vi_minus': vi_minus,
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='macd_crossover',
    feature_config=['macd'],  # Simple: just declare we need MACD features
    param_feature_mapping={
        'fast_ema': 'macd_{fast_ema}_{slow_ema}_{signal_ema}',
        'slow_ema': 'macd_{fast_ema}_{slow_ema}_{signal_ema}',
        'signal_ema': 'macd_{fast_ema}_{slow_ema}_{signal_ema}'
    }
)
def macd_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    MACD signal line crossover strategy.
    
    Returns sustained signal based on MACD vs signal line:
    - 1: MACD > Signal (bullish)
    - -1: MACD < Signal (bearish)
    - 0: Equal
    """
    fast_ema = params.get('fast_ema', 12)
    slow_ema = params.get('slow_ema', 26)
    signal_ema = params.get('signal_ema', 9)
    
    # Get features
    macd_line = features.get(f'macd_{fast_ema}_{slow_ema}_{signal_ema}_macd')
    signal_line = features.get(f'macd_{fast_ema}_{slow_ema}_{signal_ema}_signal')
    
    if macd_line is None or signal_line is None:
        return None
    
    # Determine signal
    if macd_line > signal_line:
        signal_value = 1
    elif macd_line < signal_line:
        signal_value = -1
    else:
        signal_value = 0
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')  # Default to 1m to match config
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'macd_crossover',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'fast_ema': fast_ema,              # Parameters for sparse storage separation
            'slow_ema': slow_ema,
            'signal_ema': signal_ema,
            'macd': macd_line,                 # Values for analysis
            'signal': signal_line,
            'histogram': macd_line - signal_line,
            'price': bar.get('close', 0)
        }
    }


@strategy(
    name='ichimoku_cloud_position',
    feature_config=['ichimoku'],  # Simple: just declare we need ichimoku features
    param_feature_mapping={
        'conversion_period': 'ichimoku_{conversion_period}_{base_period}',
        'base_period': 'ichimoku_{conversion_period}_{base_period}'
    }
)
def ichimoku_cloud_position(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Ichimoku cloud position strategy.
    
    Returns sustained signal based on price vs cloud position:
    - 1: Price above cloud (bullish)
    - -1: Price below cloud (bearish)
    - 0: Price in cloud (neutral)
    """
    conversion_period = params.get('conversion_period', 9)
    base_period = params.get('base_period', 26)
    
    # Get features
    span_a = features.get(f'ichimoku_{conversion_period}_{base_period}_senkou_span_a')
    span_b = features.get(f'ichimoku_{conversion_period}_{base_period}_senkou_span_b')
    price = bar.get('close', 0)
    
    if span_a is None or span_b is None:
        return None
    
    # Calculate cloud boundaries
    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)
    
    # Determine signal
    if price > cloud_top:
        signal_value = 1  # Price above cloud (bullish)
    elif price < cloud_bottom:
        signal_value = -1  # Price below cloud (bearish)
    else:
        signal_value = 0  # Price in cloud (neutral)
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')  # Default to 1m to match config
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'ichimoku_cloud_position',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'conversion_period': conversion_period,  # Parameters for sparse storage separation
            'base_period': base_period,
            'price': price,                          # Values for analysis
            'cloud_top': cloud_top,
            'cloud_bottom': cloud_bottom,
            'senkou_span_a': span_a,
            'senkou_span_b': span_b
        }
    }