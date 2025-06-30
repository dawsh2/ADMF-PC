"""
Simple momentum strategy that generates both entry and exit signals.

This strategy is designed to test signal performance calculations by
generating paired entry/exit signals based on momentum conditions.
"""

from typing import Dict, Any, Optional
from src.core.features.feature_spec import FeatureSpec
import logging
from ...core.components.discovery import strategy

logger = logging.getLogger(__name__)


@strategy(
    name='simple_momentum',
    feature_discovery=lambda params: [FeatureSpec('sma', {'period': params.get('sma_period', 20)}), FeatureSpec('rsi', {'period': params.get('rsi_period', 14)})]  # Simple list format - topology builder infers parameters
)
def simple_momentum_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Simple momentum strategy that generates entry and exit signals.
    
    Uses a simple holding period approach to ensure we get exit signals.
    
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
    exit_bars = params.get('exit_bars', 10)  # Exit after N bars
    
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
    
    # Pure stateless momentum signal generation
    # Generate signal based on current market conditions only
    signal = None
    
    # Long signal: Price above SMA and RSI oversold (momentum building up)
    if price > sma and rsi < rsi_threshold_long:
        signal = {
            'symbol': symbol,
            'direction': 'long',
            'signal_type': 'entry',
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
    
    # Short signal: Price below SMA and RSI overbought (momentum building down)
    elif price < sma and rsi > rsi_threshold_short:
        signal = {
            'symbol': symbol,
            'direction': 'short',
            'signal_type': 'entry',
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
    
    # If no strong signal, return flat (no momentum detected)
    if signal is None:
        signal = {
            'symbol': symbol,
            'direction': 'flat',
            'signal_type': 'entry',
            'strength': 0.0,
            'price': price,
            'reason': f'No momentum: price/SMA trend not confirmed by RSI',
            'indicators': {
                'price': price,
                'sma': sma,
                'rsi': rsi
            }
        }
    
    return signal