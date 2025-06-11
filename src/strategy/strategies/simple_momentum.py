"""
Simple momentum strategy without complex imports.

This is a clean implementation for testing the feature pipeline.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def simple_momentum_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Simple momentum strategy based on SMA and RSI.
    
    Args:
        features: Calculated features from FeatureHub
        bar: Current bar data
        params: Strategy parameters
        
    Returns:
        Signal dict or None
    """
    # Get parameters
    sma_period = params.get('sma_period', 20)
    rsi_threshold_long = params.get('rsi_threshold_long', 30)
    rsi_threshold_short = params.get('rsi_threshold_short', 70)
    
    # Get features
    sma = features.get(f'sma_{sma_period}') or features.get('sma_20')
    rsi = features.get('rsi')
    price = bar.get('close', 0)
    symbol = bar.get('symbol', 'UNKNOWN')
    
    logger.debug(f"Simple momentum: price={price}, sma={sma}, rsi={rsi}")
    
    # Check if we have required features
    if sma is None or rsi is None:
        logger.debug(f"Missing features for {symbol}: sma={sma}, rsi={rsi}")
        return None
    
    # Generate signal
    signal = None
    
    # Long signal: Price above SMA and RSI oversold
    if price > sma and rsi < rsi_threshold_long:
        signal = {
            'symbol': symbol,
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
        logger.info(f"Generated LONG signal for {symbol}: price={price}, sma={sma}, rsi={rsi}")
    
    # Short signal: Price below SMA and RSI overbought
    elif price < sma and rsi > rsi_threshold_short:
        signal = {
            'symbol': symbol,
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
        logger.info(f"Generated SHORT signal for {symbol}: price={price}, sma={sma}, rsi={rsi}")
    
    return signal