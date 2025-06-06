"""
Market classifiers for unified architecture.

These classifiers implement pure functions for classifying market regimes.
All functions are decorated with @classifier for discovery and use in the
event-driven architecture.
"""

from typing import Dict, Any
from enum import Enum

from ...core.containers.discovery import classifier


class MarketRegime(Enum):
    """Common market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@classifier(
    name='trend_classifier',
    regime_types=['trending_up', 'trending_down', 'ranging'],
    features=['sma_fast', 'sma_slow']
)
def trend_classifier(features: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pure function trend classifier based on moving average relationships.
    
    Args:
        features: Calculated indicators (sma_fast, sma_slow)
        params: Classifier parameters (trend_threshold)
        
    Returns:
        Regime dict with classification and confidence
    """
    # Extract parameters with defaults
    trend_threshold = params.get('trend_threshold', 0.02)
    
    # Get required features
    fast_ma = features.get('sma_fast')
    slow_ma = features.get('sma_slow')
    
    # Return default if features missing
    if fast_ma is None or slow_ma is None or slow_ma == 0:
        return {
            'regime': 'ranging',
            'confidence': 0.0,
            'metadata': {'reason': 'Missing required features'}
        }
    
    # Calculate trend strength
    ma_diff = (fast_ma - slow_ma) / slow_ma
    
    # Classify based on MA relationship
    if ma_diff > trend_threshold:
        return {
            'regime': 'trending_up',
            'confidence': min(ma_diff / (trend_threshold * 2), 1.0),
            'metadata': {
                'ma_diff': ma_diff,
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'reason': 'Fast MA above slow MA'
            }
        }
    elif ma_diff < -trend_threshold:
        return {
            'regime': 'trending_down',
            'confidence': min(abs(ma_diff) / (trend_threshold * 2), 1.0),
            'metadata': {
                'ma_diff': ma_diff,
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'reason': 'Fast MA below slow MA'
            }
        }
    else:
        return {
            'regime': 'ranging',
            'confidence': 1.0 - (abs(ma_diff) / trend_threshold),
            'metadata': {
                'ma_diff': ma_diff,
                'reason': 'No clear trend'
            }
        }


@classifier(
    name='volatility_classifier',
    regime_types=['high_volatility', 'low_volatility', 'normal_volatility'],
    features=['atr', 'atr_sma', 'volatility', 'volatility_sma']
)
def volatility_classifier(features: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pure function volatility classifier based on ATR or standard deviation.
    
    Args:
        features: Calculated indicators (atr, atr_sma, or volatility measures)
        params: Classifier parameters (high_vol_threshold, low_vol_threshold)
        
    Returns:
        Regime dict with classification and confidence
    """
    # Extract parameters with defaults
    high_vol_threshold = params.get('high_vol_threshold', 1.5)
    low_vol_threshold = params.get('low_vol_threshold', 0.5)
    
    # Get volatility measure (prefer ATR if available)
    current_vol = features.get('atr') or features.get('volatility')
    avg_vol = features.get('atr_sma') or features.get('volatility_sma')
    
    # Return default if features missing
    if current_vol is None or avg_vol is None or avg_vol == 0:
        return {
            'regime': 'normal_volatility',
            'confidence': 0.0,
            'metadata': {'reason': 'Missing volatility features'}
        }
    
    # Calculate volatility ratio
    vol_ratio = current_vol / avg_vol
    
    # Classify based on volatility level
    if vol_ratio > high_vol_threshold:
        return {
            'regime': 'high_volatility',
            'confidence': min((vol_ratio - 1.0) / (high_vol_threshold - 1.0), 1.0),
            'metadata': {
                'vol_ratio': vol_ratio,
                'current_vol': current_vol,
                'avg_vol': avg_vol,
                'reason': 'Volatility above average'
            }
        }
    elif vol_ratio < low_vol_threshold:
        return {
            'regime': 'low_volatility',
            'confidence': min((1.0 - vol_ratio) / (1.0 - low_vol_threshold), 1.0),
            'metadata': {
                'vol_ratio': vol_ratio,
                'current_vol': current_vol,
                'avg_vol': avg_vol,
                'reason': 'Volatility below average'
            }
        }
    else:
        return {
            'regime': 'normal_volatility',
            'confidence': 0.5,
            'metadata': {
                'vol_ratio': vol_ratio,
                'reason': 'Normal volatility'
            }
        }


@classifier(
    name='momentum_regime_classifier',
    regime_types=['strong_momentum', 'weak_momentum', 'no_momentum'],
    features=['rsi', 'macd', 'momentum']
)
def momentum_regime_classifier(features: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pure function momentum regime classifier based on RSI and momentum indicators.
    
    Args:
        features: Calculated indicators (rsi, macd, momentum)
        params: Classifier parameters (rsi_overbought, rsi_oversold)
        
    Returns:
        Regime dict with classification and confidence
    """
    # Extract parameters
    rsi_overbought = params.get('rsi_overbought', 70)
    rsi_oversold = params.get('rsi_oversold', 30)
    momentum_threshold = params.get('momentum_threshold', 0.02)
    
    # Get features
    rsi = features.get('rsi')
    momentum = features.get('momentum', 0)
    macd = features.get('macd')
    
    if rsi is None:
        return {
            'regime': 'no_momentum',
            'confidence': 0.0,
            'metadata': {'reason': 'Missing RSI'}
        }
    
    # Calculate momentum strength
    if rsi > rsi_overbought or rsi < rsi_oversold:
        strength = abs(rsi - 50) / 50
        direction = 'bullish' if rsi > 50 else 'bearish'
        
        return {
            'regime': 'strong_momentum',
            'confidence': strength,
            'metadata': {
                'rsi': rsi,
                'momentum': momentum,
                'macd': macd,
                'direction': direction,
                'reason': f'RSI in {direction} extreme'
            }
        }
    elif abs(momentum) > momentum_threshold:
        return {
            'regime': 'weak_momentum',
            'confidence': min(abs(momentum) / (momentum_threshold * 2), 1.0),
            'metadata': {
                'rsi': rsi,
                'momentum': momentum,
                'direction': 'bullish' if momentum > 0 else 'bearish',
                'reason': 'Momentum present but RSI neutral'
            }
        }
    else:
        return {
            'regime': 'no_momentum',
            'confidence': 0.8,
            'metadata': {
                'rsi': rsi,
                'momentum': momentum,
                'reason': 'No significant momentum'
            }
        }