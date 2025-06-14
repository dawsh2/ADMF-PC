"""
Pure MA crossover strategy for testing signal generation.

This strategy generates signals purely based on MA position,
not crossovers, so it should generate a signal on every bar
after the warmup period.
"""

from typing import Dict, Any, Optional
import logging

from ...core.components.discovery import strategy

logger = logging.getLogger(__name__)


@strategy(
    name='ma_crossover',
    feature_config={
        'sma': {'params': ['fast_period', 'slow_period'], 'defaults': {'fast_period': 5, 'slow_period': 20}}
    },
    validate_features=False
)
def ma_crossover_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Simple MA position strategy - generates signal based on fast MA vs slow MA.
    
    Args:
        features: Calculated features (should include SMAs)
        bar: Current bar data
        params: Strategy parameters
        
    Returns:
        Signal dict or None
    """
    # Get parameters
    fast_period = params.get('fast_period', 5)
    slow_period = params.get('slow_period', 20)
    
    # Get features
    fast_sma = features.get(f'sma_{fast_period}')
    slow_sma = features.get(f'sma_{slow_period}')
    price = bar.get('close', 0)
    symbol = bar.get('symbol', 'UNKNOWN')
    
    logger.debug(f"MA Crossover called with {len(features)} features (fast_sma={fast_sma}, slow_sma={slow_sma})")
    logger.info(f"MA Crossover: price={price}, fast_sma={fast_sma}, slow_sma={slow_sma}")
    
    # Check if we have required features
    if fast_sma is None or slow_sma is None:
        if fast_sma is None and slow_sma is None:
            logger.debug(f"Insufficient data for {symbol}: Neither SMA available yet")
        elif slow_sma is None:
            logger.debug(f"Insufficient data for {symbol}: {slow_period}-period SMA not yet available (have {fast_period}-period)")
        else:
            logger.debug(f"Insufficient data for {symbol}: {fast_period}-period SMA not available")
        return None
    
    # Generate signal based on MA position (using fixed values for sparse storage)
    if fast_sma > slow_sma:
        signal = {
            'symbol': symbol,
            'direction': 'long',
            'signal_type': 'entry',
            'strength': 1.0,  # Fixed value: 1 for long
            'price': price
        }
        logger.info(f"Generated LONG signal for {symbol}: fast_sma={fast_sma:.2f} > slow_sma={slow_sma:.2f}")
    else:
        signal = {
            'symbol': symbol,
            'direction': 'short', 
            'signal_type': 'entry',
            'strength': -1.0,  # Fixed value: -1 for short
            'price': price
        }
        logger.info(f"Generated SHORT signal for {symbol}: fast_sma={fast_sma:.2f} < slow_sma={slow_sma:.2f}")
    
    return signal