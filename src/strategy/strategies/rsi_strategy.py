"""
RSI Strategy

Simple RSI-based strategy that trades on oversold/overbought conditions.
"""

import logging
from typing import Dict, Any, Optional

from ...core.components.discovery import strategy

logger = logging.getLogger(__name__)


@strategy(
    name='rsi_strategy',
    feature_config=['rsi']  # Topology builder infers parameters from strategy logic
)
def rsi_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    RSI-based trading strategy.
    
    Entry:
    - Long when RSI < oversold_threshold
    - Short when RSI > overbought_threshold
    
    Exit:
    - When RSI crosses back above/below thresholds
    """
    logger.info(f"RSI_STRATEGY CALLED: features={list(features.keys())}, bar_close={bar.get('close')}, params={params}")
    # Parameters
    rsi_period = params.get('rsi_period', 14)
    oversold_threshold = params.get('oversold_threshold', 30)
    overbought_threshold = params.get('overbought_threshold', 70)
    
    # Get RSI value
    rsi = features.get(f'rsi_{rsi_period}', features.get('rsi'))
    
    if rsi is None:
        return None
    
    # Get symbol from bar
    symbol = bar.get('symbol', 'UNKNOWN')
    current_price = bar.get('close', 0)
    
    # Generate signal based on RSI
    if rsi < oversold_threshold:
        signal = {
            'symbol': symbol,
            'direction': 'long',
            'signal_type': 'entry',
            'strength': min((oversold_threshold - rsi) / oversold_threshold, 1.0),
            'price': current_price,
            'reason': f'RSI oversold: {rsi:.1f} < {oversold_threshold}',
            'indicators': {
                'rsi': rsi,
                'oversold_threshold': oversold_threshold
            }
        }
    elif rsi > overbought_threshold:
        signal = {
            'symbol': symbol,
            'direction': 'short',
            'signal_type': 'entry', 
            'strength': min((rsi - overbought_threshold) / (100 - overbought_threshold), 1.0),
            'price': current_price,
            'reason': f'RSI overbought: {rsi:.1f} > {overbought_threshold}',
            'indicators': {
                'rsi': rsi,
                'overbought_threshold': overbought_threshold
            }
        }
    else:
        # Neutral signal
        signal = {
            'symbol': symbol,
            'direction': 'flat',
            'signal_type': 'entry',
            'strength': 0.0,
            'price': current_price,
            'reason': f'RSI neutral: {rsi:.1f}',
            'indicators': {
                'rsi': rsi
            }
        }
    
    return signal