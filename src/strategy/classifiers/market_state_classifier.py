"""
Market State Classifier

Classifies market into different states based on volatility and trend.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
from ...core.components.discovery import classifier

logger = logging.getLogger(__name__)


@classifier(
    name='market_state_classifier',
    regime_types=['trending_low_vol', 'trending_high_vol', 'ranging_low_vol', 'ranging_high_vol'],
    feature_config=['atr', 'sma'],
    param_feature_mapping={
        'vol_lookback': 'atr_{vol_lookback}',
        'trend_lookback': 'sma_{trend_lookback}'
    },
    parameter_space={
        'vol_lookback': {'type': 'int', 'range': (10, 50), 'default': 20},
        'trend_lookback': {'type': 'int', 'range': (20, 100), 'default': 50},
        'regime_threshold': {'type': 'float', 'range': (0.3, 0.7), 'default': 0.5}
    }
)
def market_state_classifier(
    features: Dict[str, float],
    params: Dict[str, Any]
) -> Dict[str, Any]:
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
        return {
            'regime': 'ranging_low_vol',
            'confidence': 0.0,
            'metadata': {'reason': 'Missing ATR or price data'}
        }
    
    # Normalize volatility as percentage of price
    vol_pct = (atr / price) * 100
    
    # Get trend strength from moving averages
    fast_ma = features.get(f'sma_{trend_lookback//2}')
    slow_ma = features.get(f'sma_{trend_lookback}')
    
    if fast_ma is None or slow_ma is None:
        return {
            'regime': 'ranging_low_vol',
            'confidence': 0.0,
            'metadata': {'reason': 'Missing moving average data'}
        }
    
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
            confidence = min(trend_strength / (regime_threshold * 2) + vol_pct / (regime_threshold * 2), 1.0)
        else:
            regime = "trending_low_vol"
            confidence = trend_strength / (regime_threshold * 2) * (1 - vol_pct / regime_threshold)
    else:
        if is_high_vol:
            regime = "ranging_high_vol"
            confidence = vol_pct / (regime_threshold * 2) * (1 - trend_strength / regime_threshold)
        else:
            regime = "ranging_low_vol"
            confidence = (1 - trend_strength / regime_threshold) * (1 - vol_pct / regime_threshold)
    
    confidence = max(0.0, min(confidence, 1.0))  # Clamp to [0, 1]
    
    logger.debug(f"Market state: {regime} (vol={vol_pct:.2f}%, trend={trend_strength:.2f}%)")
    
    return {
        'regime': regime,
        'confidence': confidence,
        'metadata': {
            'vol_pct': vol_pct,
            'trend_strength': trend_strength,
            'is_high_vol': is_high_vol,
            'is_trending': is_trending,
            'reason': f'Vol: {vol_pct:.2f}%, Trend: {trend_strength:.2f}%'
        }
    }