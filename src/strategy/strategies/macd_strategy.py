"""
MACD Strategy

MACD crossover strategy with signal line.
"""

import logging
from src.core.features.feature_spec import FeatureSpec
from typing import Dict, Any, Optional

from ...core.components.discovery import strategy

logger = logging.getLogger(__name__)


@strategy(
    name='macd_strategy',
    feature_discovery=lambda params: [FeatureSpec('macd', {'fast_period': params.get('fast_period', 12), 'slow_period': params.get('slow_period', 26), 'signal_period': params.get('signal_period', 9)})]  # Topology builder infers parameters from strategy logic
)
def macd_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    MACD crossover strategy.
    
    Entry:
    - Long when MACD crosses above signal line
    - Short when MACD crosses below signal line
    
    Exit:
    - When opposite crossover occurs
    """
    # Parameters
    fast_ema = params.get('fast_ema', 12)
    slow_ema = params.get('slow_ema', 26)
    signal_ema = params.get('signal_ema', 9)
    threshold = params.get('threshold', 0.0)
    
    # Get MACD values
    macd_line = features.get('macd_line')
    signal_line = features.get('macd_signal')
    macd_hist = features.get('macd_histogram')
    
    if macd_line is None or signal_line is None:
        return None
    
    # Calculate histogram if not provided
    if macd_hist is None:
        macd_hist = macd_line - signal_line
    
    # Get symbol and price from bar
    symbol = bar.get('symbol', 'UNKNOWN')
    current_price = bar.get('close', 0)
    
    # Simple crossover detection based on histogram
    if macd_hist > threshold:
        signal = {
            'symbol': symbol,
            'direction': 'long',
            'signal_type': 'entry',
            'strength': min(abs(macd_hist) / 0.01, 1.0),  # Normalize by typical histogram value
            'price': current_price,
            'reason': f'MACD bullish: histogram={macd_hist:.4f}',
            'indicators': {
                'macd_line': macd_line,
                'signal_line': signal_line,
                'macd_histogram': macd_hist
            }
        }
    elif macd_hist < -threshold:
        signal = {
            'symbol': symbol,
            'direction': 'short',
            'signal_type': 'entry',
            'strength': min(abs(macd_hist) / 0.01, 1.0),
            'price': current_price,
            'reason': f'MACD bearish: histogram={macd_hist:.4f}',
            'indicators': {
                'macd_line': macd_line,
                'signal_line': signal_line,
                'macd_histogram': macd_hist
            }
        }
    else:
        signal = {
            'symbol': symbol,
            'direction': 'flat',
            'signal_type': 'entry',
            'strength': 0.0,
            'price': current_price,
            'reason': f'MACD neutral: histogram={macd_hist:.4f}',
            'indicators': {
                'macd_line': macd_line,
                'signal_line': signal_line,
                'macd_histogram': macd_hist
            }
        }
    
    return signal