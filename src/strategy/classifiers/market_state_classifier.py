"""
Market State Classifier

Classifies market into different states based on volatility and trend.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


def market_state_classifier(
    symbol: str,
    features: Dict[str, float],
    params: Dict[str, Any]
) -> Optional[str]:
    """
    Classify market state based on volatility and trend.
    
    States:
    - trending_low_vol: Strong trend with low volatility
    - trending_high_vol: Strong trend with high volatility
    - ranging_low_vol: Range-bound with low volatility
    - ranging_high_vol: Range-bound with high volatility (choppy)
    
    Parameters:
    - vol_lookback: Period for volatility calculation
    - trend_lookback: Period for trend strength calculation
    - regime_threshold: Threshold for regime classification
    """
    # Parameters
    vol_lookback = params.get('vol_lookback', 20)
    trend_lookback = params.get('trend_lookback', 50)
    regime_threshold = params.get('regime_threshold', 0.5)
    
    # Get volatility measure
    atr = features.get(f'atr_{vol_lookback}', features.get('atr'))
    price = features.get('close', 0)
    
    if atr is None or price <= 0:
        return None
    
    # Normalize volatility as percentage of price
    vol_pct = (atr / price) * 100
    
    # Get trend strength from moving averages
    fast_ma = features.get(f'sma_{trend_lookback//2}')
    slow_ma = features.get(f'sma_{trend_lookback}')
    
    if fast_ma is None or slow_ma is None:
        return None
    
    # Calculate trend strength
    trend_strength = abs(fast_ma - slow_ma) / slow_ma * 100
    
    # Classify volatility
    is_high_vol = vol_pct > regime_threshold
    
    # Classify trend
    is_trending = trend_strength > regime_threshold
    
    # Determine market state
    if is_trending:
        if is_high_vol:
            regime = "trending_high_vol"
        else:
            regime = "trending_low_vol"
    else:
        if is_high_vol:
            regime = "ranging_high_vol"
        else:
            regime = "ranging_low_vol"
    
    logger.debug(f"{symbol} market state: {regime} (vol={vol_pct:.2f}%, trend={trend_strength:.2f}%)")
    
    return regime