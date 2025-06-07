"""
Momentum Strategy for EVENT_FLOW_ARCHITECTURE

This module contains pure functions that generate trading signals based on momentum indicators.
No state, no inheritance - just functions that take features and return signals.
"""

from typing import Dict, Any, Optional
import logging

from ...core.components.discovery import strategy

logger = logging.getLogger(__name__)


@strategy(
    feature_config={
        'sma': {'params': ['fast_period', 'slow_period'], 'default': 20},
        'rsi': {'params': ['rsi_period'], 'default': 14}
    }
)
def momentum_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generate momentum-based trading signals.
    
    This is a pure function - no state maintained between calls.
    
    Args:
        features: Dictionary of computed features (SMA, RSI, etc.)
        bar: Current market bar data
        params: Strategy parameters
        
    Returns:
        Signal dictionary or None if no signal
    """
    # Extract parameters
    sma_period = params.get('sma_period', 20)
    rsi_threshold_long = params.get('rsi_threshold_long', 30)
    rsi_threshold_short = params.get('rsi_threshold_short', 70)
    
    logger.debug(f"Momentum strategy called with features: {list(features.keys())}, params: {params}")
    
    # Get required features
    price = bar.get('close', 0)
    sma_key = f'sma_{sma_period}'
    sma = features.get(sma_key)
    rsi = features.get('rsi')
    
    # Check if we have required features
    if sma is None or rsi is None:
        logger.debug(f"Missing features: sma_{sma_period}={sma}, rsi={rsi}")
        return None
    
    # Generate signal based on momentum logic
    signal = None
    
    # Long signal: Price above SMA and RSI oversold
    if price > sma and rsi < rsi_threshold_long:
        signal = {
            'symbol': bar.get('symbol'),
            'direction': 'long',
            'strength': min(1.0, (rsi_threshold_long - rsi) / rsi_threshold_long),
            'price': price,
            'reason': f'Momentum long: price > SMA{sma_period} and RSI < {rsi_threshold_long}',
            'indicators': {
                'price': price,
                'sma': sma,
                'rsi': rsi
            }
        }
        logger.info(f"Generated LONG signal: price={price}, sma={sma}, rsi={rsi}")
    
    # Short signal: Price below SMA and RSI overbought
    elif price < sma and rsi > rsi_threshold_short:
        signal = {
            'symbol': bar.get('symbol'),
            'direction': 'short',
            'strength': min(1.0, (rsi - rsi_threshold_short) / (100 - rsi_threshold_short)),
            'price': price,
            'reason': f'Momentum short: price < SMA{sma_period} and RSI > {rsi_threshold_short}',
            'indicators': {
                'price': price,
                'sma': sma,
                'rsi': rsi
            }
        }
        logger.info(f"Generated SHORT signal: price={price}, sma={sma}, rsi={rsi}")
    
    return signal


# Alternative momentum strategies with different logic

@strategy(
    name='dual_momentum',
    indicators={
        'sma': {'params': ['fast_period', 'slow_period'], 'defaults': {'fast_period': 10, 'slow_period': 30}}
    }
)
def dual_momentum_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Dual momentum strategy using fast and slow moving averages.
    """
    # Extract parameters
    fast_period = params.get('fast_period', 10)
    slow_period = params.get('slow_period', 30)
    
    # Get features
    price = bar.get('close', 0)
    fast_sma = features.get(f'sma_{fast_period}')
    slow_sma = features.get(f'sma_{slow_period}')
    
    if fast_sma is None or slow_sma is None:
        return None
    
    # Generate signal based on crossover
    signal = None
    
    # Long signal: Fast MA crosses above slow MA
    if fast_sma > slow_sma and price > fast_sma:
        signal = {
            'symbol': bar.get('symbol'),
            'direction': 'long',
            'strength': min(1.0, (fast_sma - slow_sma) / slow_sma),
            'price': price,
            'reason': f'Dual momentum long: SMA{fast_period} > SMA{slow_period}',
            'indicators': {
                'price': price,
                'fast_sma': fast_sma,
                'slow_sma': slow_sma
            }
        }
    
    # Short signal: Fast MA crosses below slow MA
    elif fast_sma < slow_sma and price < fast_sma:
        signal = {
            'symbol': bar.get('symbol'),
            'direction': 'short',
            'strength': min(1.0, (slow_sma - fast_sma) / slow_sma),
            'price': price,
            'reason': f'Dual momentum short: SMA{fast_period} < SMA{slow_period}',
            'indicators': {
                'price': price,
                'fast_sma': fast_sma,
                'slow_sma': slow_sma
            }
        }
    
    return signal


@strategy(
    name='price_momentum',
    indicators={
        'price_history': {'params': ['lookback_period'], 'default': 20}
    }
)
def price_momentum_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Simple price momentum based on rate of change.
    """
    # Extract parameters
    lookback = params.get('lookback_period', 20)
    threshold = params.get('momentum_threshold', 0.02)  # 2% threshold
    
    # Calculate price momentum
    price = bar.get('close', 0)
    lookback_price = features.get(f'close_{lookback}')  # Price N bars ago
    
    if lookback_price is None or lookback_price == 0:
        return None
    
    # Calculate momentum
    momentum = (price - lookback_price) / lookback_price
    
    # Generate signal
    signal = None
    
    if momentum > threshold:
        signal = {
            'symbol': bar.get('symbol'),
            'direction': 'long',
            'strength': min(1.0, momentum / (threshold * 2)),
            'price': price,
            'reason': f'Price momentum long: {momentum:.2%} over {lookback} bars',
            'indicators': {
                'price': price,
                'lookback_price': lookback_price,
                'momentum': momentum
            }
        }
    elif momentum < -threshold:
        signal = {
            'symbol': bar.get('symbol'),
            'direction': 'short',
            'strength': min(1.0, abs(momentum) / (threshold * 2)),
            'price': price,
            'reason': f'Price momentum short: {momentum:.2%} over {lookback} bars',
            'indicators': {
                'price': price,
                'lookback_price': lookback_price,
                'momentum': momentum
            }
        }
    
    return signal