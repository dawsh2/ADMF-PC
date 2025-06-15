"""
Market classifiers for unified architecture.

These classifiers implement pure functions for classifying market regimes.
All functions are decorated with @classifier for discovery and use in the
event-driven architecture.
"""

from typing import Dict, Any
from enum import Enum

from ...core.components.discovery import classifier


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
    feature_config=['sma'],
    param_feature_mapping={
        'fast_period': 'sma_{fast_period}',
        'slow_period': 'sma_{slow_period}'
    }
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
    fast_period = params.get('fast_period', 10)
    slow_period = params.get('slow_period', 20)
    
    # Get required features - try multiple naming patterns
    fast_ma = features.get('sma_fast') or features.get(f'sma_{fast_period}')
    slow_ma = features.get('sma_slow') or features.get(f'sma_{slow_period}')
    
    # Also check for strategy-specific feature names
    if fast_ma is None or slow_ma is None:
        # Look for any SMA features and use appropriate ones based on period
        sma_features = {k: v for k, v in features.items() if k.startswith('sma_')}
        if sma_features:
            sorted_smas = sorted([(int(k.split('_')[1]), v) for k, v in sma_features.items() if '_' in k and k.split('_')[1].isdigit()])
            if len(sorted_smas) >= 2:
                # Use smallest period as fast, larger as slow
                fast_ma = sorted_smas[0][1]
                slow_ma = sorted_smas[-1][1]
    
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
    feature_config=['atr'],
    param_feature_mapping={
        'atr_period': 'atr_{atr_period}',
        'lookback_period': 'atr_{lookback_period}'
    }
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
    # Extract parameters with defaults - support both atr_period and lookback_period
    high_vol_threshold = params.get('high_vol_threshold', 1.5)
    low_vol_threshold = params.get('low_vol_threshold', 0.5)
    atr_period = params.get('atr_period') or params.get('lookback_period', 14)
    
    # Get volatility measure - try multiple naming patterns
    current_vol = features.get('atr') or features.get(f'atr_{atr_period}') or features.get('volatility')
    avg_vol = features.get('atr_sma') or features.get(f'atr_{atr_period}_sma') or features.get('volatility_sma')
    
    # If no volatility features found, try to compute from price movement
    if current_vol is None and 'close' in features:
        # Use price-based volatility as fallback
        price = features.get('close', 0)
        if price > 0:
            # Look for any high-low range
            high = features.get('high', price)
            low = features.get('low', price) 
            current_vol = (high - low) / price if price > 0 else 0
            avg_vol = current_vol  # Use same value if no average available
    
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
    feature_config=['rsi', 'momentum', 'macd'],
    param_feature_mapping={
        'rsi_period': 'rsi_{rsi_period}',
        'momentum_period': 'momentum_{momentum_period}',
        'macd_fast': 'macd_{macd_fast}_{macd_slow}_{macd_signal}',
        'macd_slow': 'macd_{macd_fast}_{macd_slow}_{macd_signal}',
        'macd_signal': 'macd_{macd_fast}_{macd_slow}_{macd_signal}'
    }
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
    rsi_period = params.get('rsi_period', 14)
    momentum_period = params.get('momentum_period', 10)
    
    # Get features - try multiple naming patterns
    rsi = features.get('rsi') or features.get(f'rsi_{rsi_period}')
    
    # Try various momentum feature names
    momentum = (features.get('momentum_momentum_10') or 
                features.get(f'momentum_momentum_{momentum_period}') or
                features.get('momentum_momentum_20') or 
                features.get('momentum') or
                features.get(f'momentum_{momentum_period}') or
                0)
    
    # Try various MACD feature names
    macd = features.get('macd_macd') or features.get('macd')
    
    # If no momentum indicators, try to compute from price changes
    if rsi is None and momentum == 0:
        # Look for price-based momentum
        close = features.get('close')
        if close and any(k.startswith('sma_') for k in features):
            # Use SMA crossover as momentum proxy
            sma_features = {k: v for k, v in features.items() if k.startswith('sma_')}
            if len(sma_features) >= 2:
                sorted_smas = sorted(sma_features.values())
                momentum = (sorted_smas[-1] - sorted_smas[0]) / sorted_smas[0] if sorted_smas[0] > 0 else 0
    
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


