"""
Enhanced Multi-State Classifiers

These classifiers implement more granular regime classification with 3-5 states
for better regime-aware strategy analysis.
"""

from typing import Dict, Any
from ...core.components.discovery import classifier


@classifier(
    name='enhanced_trend_classifier',
    regime_types=['strong_uptrend', 'weak_uptrend', 'sideways', 'weak_downtrend', 'strong_downtrend'],
    features=['sma_10', 'sma_20', 'sma_50']
)
def enhanced_trend_classifier(features: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Five-state trend classifier using multiple timeframe analysis.
    
    States:
    - strong_uptrend: All MAs aligned upward, strong momentum
    - weak_uptrend: MAs mostly upward, weak momentum  
    - sideways: MAs converging, no clear direction
    - weak_downtrend: MAs mostly downward, weak momentum
    - strong_downtrend: All MAs aligned downward, strong momentum
    """
    # Parameters
    strong_threshold = params.get('strong_threshold', 0.02)  # 2%
    weak_threshold = params.get('weak_threshold', 0.005)    # 0.5%
    
    # Get moving averages
    sma_10 = features.get('sma_10')
    sma_20 = features.get('sma_20') 
    sma_50 = features.get('sma_50')
    price = features.get('close', 0)
    
    if not all([sma_10, sma_20, sma_50, price > 0]):
        return {
            'regime': 'sideways',
            'confidence': 0.0,
            'metadata': {'reason': 'Missing features'}
        }
    
    # Calculate trend gradients
    short_trend = (sma_10 - sma_20) / sma_20
    medium_trend = (sma_20 - sma_50) / sma_50
    price_trend = (price - sma_10) / sma_10
    
    # Average trend strength
    avg_trend = (short_trend + medium_trend + price_trend) / 3
    
    # Classification logic
    if avg_trend > strong_threshold:
        return {
            'regime': 'strong_uptrend',
            'confidence': min(avg_trend / (strong_threshold * 2), 1.0),
            'metadata': {
                'avg_trend': avg_trend,
                'short_trend': short_trend,
                'medium_trend': medium_trend,
                'price_trend': price_trend,
                'reason': 'Strong bullish alignment'
            }
        }
    elif avg_trend > weak_threshold:
        return {
            'regime': 'weak_uptrend',
            'confidence': avg_trend / strong_threshold,
            'metadata': {
                'avg_trend': avg_trend,
                'reason': 'Weak bullish trend'
            }
        }
    elif avg_trend < -strong_threshold:
        return {
            'regime': 'strong_downtrend',
            'confidence': min(abs(avg_trend) / (strong_threshold * 2), 1.0),
            'metadata': {
                'avg_trend': avg_trend,
                'short_trend': short_trend,
                'medium_trend': medium_trend,
                'price_trend': price_trend,
                'reason': 'Strong bearish alignment'
            }
        }
    elif avg_trend < -weak_threshold:
        return {
            'regime': 'weak_downtrend',
            'confidence': abs(avg_trend) / strong_threshold,
            'metadata': {
                'avg_trend': avg_trend,
                'reason': 'Weak bearish trend'
            }
        }
    else:
        return {
            'regime': 'sideways',
            'confidence': 1.0 - (abs(avg_trend) / weak_threshold),
            'metadata': {
                'avg_trend': avg_trend,
                'reason': 'No clear trend'
            }
        }


@classifier(
    name='volatility_momentum_classifier',
    regime_types=['high_vol_bullish', 'high_vol_bearish', 'low_vol_bullish', 'low_vol_bearish', 'neutral'],
    features=['atr_14', 'rsi_14', 'sma_20']
)
def volatility_momentum_classifier(features: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Five-state classifier combining volatility and momentum.
    
    States:
    - high_vol_bullish: High volatility with bullish momentum
    - high_vol_bearish: High volatility with bearish momentum
    - low_vol_bullish: Low volatility with bullish momentum
    - low_vol_bearish: Low volatility with bearish momentum
    - neutral: Normal volatility and momentum
    """
    # Parameters
    vol_threshold = params.get('vol_threshold', 1.5)
    rsi_overbought = params.get('rsi_overbought', 65)
    rsi_oversold = params.get('rsi_oversold', 35)
    
    # Get features
    atr = features.get('atr_14') or features.get('atr')
    rsi = features.get('rsi_14') or features.get('rsi')
    sma = features.get('sma_20')
    price = features.get('close', 0)
    
    if not all([atr, rsi, sma, price > 0]):
        return {
            'regime': 'neutral',
            'confidence': 0.0,
            'metadata': {'reason': 'Missing features'}
        }
    
    # Calculate volatility level (relative to price)
    vol_pct = (atr / price) * 100
    is_high_vol = vol_pct > vol_threshold
    
    # Determine momentum direction
    price_momentum = (price - sma) / sma
    is_bullish = rsi > 50 and price_momentum > 0
    is_bearish = rsi < 50 and price_momentum < 0
    
    # Classification
    if is_high_vol:
        if is_bullish:
            regime = 'high_vol_bullish'
            confidence = min((rsi - 50) / 50 + vol_pct / (vol_threshold * 2), 1.0)
        elif is_bearish:
            regime = 'high_vol_bearish' 
            confidence = min((50 - rsi) / 50 + vol_pct / (vol_threshold * 2), 1.0)
        else:
            regime = 'neutral'
            confidence = 0.5
    else:
        if is_bullish:
            regime = 'low_vol_bullish'
            confidence = (rsi - 50) / 50 * (1 - vol_pct / vol_threshold)
        elif is_bearish:
            regime = 'low_vol_bearish'
            confidence = (50 - rsi) / 50 * (1 - vol_pct / vol_threshold)
        else:
            regime = 'neutral'
            confidence = 0.8
    
    return {
        'regime': regime,
        'confidence': confidence,
        'metadata': {
            'vol_pct': vol_pct,
            'rsi': rsi,
            'price_momentum': price_momentum,
            'is_high_vol': is_high_vol,
            'is_bullish': is_bullish,
            'is_bearish': is_bearish,
            'reason': f'Vol: {vol_pct:.2f}%, RSI: {rsi:.1f}, Price vs SMA: {price_momentum:.3f}'
        }
    }


@classifier(
    name='market_regime_classifier',
    regime_types=['bull_trending', 'bull_ranging', 'bear_trending', 'bear_ranging', 'neutral'],
    features=['sma_10', 'sma_50', 'atr_20', 'rsi_14']
)
def market_regime_classifier(features: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Five-state market regime classifier.
    
    States:
    - bull_trending: Bullish with clear trend
    - bull_ranging: Bullish but sideways
    - bear_trending: Bearish with clear trend  
    - bear_ranging: Bearish but sideways
    - neutral: No clear bias
    """
    # Parameters
    trend_threshold = params.get('trend_threshold', 0.01)
    vol_threshold = params.get('vol_threshold', 1.0)
    
    # Get features
    sma_short = features.get('sma_10')
    sma_long = features.get('sma_50')
    atr = features.get('atr_20') or features.get('atr')
    rsi = features.get('rsi_14') or features.get('rsi')
    price = features.get('close', 0)
    
    if not all([sma_short, sma_long, atr, rsi, price > 0]):
        return {
            'regime': 'neutral',
            'confidence': 0.0,
            'metadata': {'reason': 'Missing features'}
        }
    
    # Calculate trend and volatility
    trend_strength = (sma_short - sma_long) / sma_long
    vol_level = (atr / price) * 100
    
    # Determine market bias
    is_bullish = trend_strength > 0 and rsi > 50
    is_bearish = trend_strength < 0 and rsi < 50
    is_trending = abs(trend_strength) > trend_threshold and vol_level > vol_threshold
    
    # Classification
    if is_bullish:
        if is_trending:
            regime = 'bull_trending'
            confidence = min(trend_strength / (trend_threshold * 2) + (rsi - 50) / 50, 1.0)
        else:
            regime = 'bull_ranging'
            confidence = (rsi - 50) / 50 * (1 - abs(trend_strength) / trend_threshold)
    elif is_bearish:
        if is_trending:
            regime = 'bear_trending'
            confidence = min(abs(trend_strength) / (trend_threshold * 2) + (50 - rsi) / 50, 1.0)
        else:
            regime = 'bear_ranging'
            confidence = (50 - rsi) / 50 * (1 - abs(trend_strength) / trend_threshold)
    else:
        regime = 'neutral'
        confidence = 1 - abs(rsi - 50) / 50
    
    return {
        'regime': regime,
        'confidence': confidence,
        'metadata': {
            'trend_strength': trend_strength,
            'vol_level': vol_level,
            'rsi': rsi,
            'is_bullish': is_bullish,
            'is_bearish': is_bearish,
            'is_trending': is_trending,
            'reason': f'Trend: {trend_strength:.3f}, Vol: {vol_level:.2f}%, RSI: {rsi:.1f}'
        }
    }


@classifier(
    name='microstructure_classifier', 
    regime_types=['breakout_up', 'breakout_down', 'consolidation', 'reversal_up', 'reversal_down'],
    features=['sma_5', 'sma_20', 'atr_10', 'rsi_7']
)
def microstructure_classifier(features: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Five-state microstructure classifier for short-term patterns.
    
    States:
    - breakout_up: Price breaking above resistance with momentum
    - breakout_down: Price breaking below support with momentum
    - consolidation: Price trading in tight range
    - reversal_up: Oversold bounce pattern
    - reversal_down: Overbought pullback pattern
    """
    # Parameters
    breakout_threshold = params.get('breakout_threshold', 0.005)  # 0.5%
    consolidation_threshold = params.get('consolidation_threshold', 0.002)  # 0.2%
    
    # Get features
    sma_fast = features.get('sma_5')
    sma_slow = features.get('sma_20')
    atr = features.get('atr_10') or features.get('atr')
    rsi = features.get('rsi_7') or features.get('rsi')
    price = features.get('close', 0)
    
    if not all([sma_fast, sma_slow, atr, rsi, price > 0]):
        return {
            'regime': 'consolidation',
            'confidence': 0.0,
            'metadata': {'reason': 'Missing features'}
        }
    
    # Calculate metrics
    price_vs_fast = (price - sma_fast) / sma_fast
    fast_vs_slow = (sma_fast - sma_slow) / sma_slow
    vol_pct = (atr / price) * 100
    
    # Classification logic
    if abs(fast_vs_slow) < consolidation_threshold and vol_pct < 0.5:
        return {
            'regime': 'consolidation',
            'confidence': 1 - abs(fast_vs_slow) / consolidation_threshold,
            'metadata': {
                'fast_vs_slow': fast_vs_slow,
                'vol_pct': vol_pct,
                'reason': 'Tight consolidation pattern'
            }
        }
    
    if price_vs_fast > breakout_threshold and rsi > 60:
        return {
            'regime': 'breakout_up',
            'confidence': min(price_vs_fast / (breakout_threshold * 2) + (rsi - 50) / 50, 1.0),
            'metadata': {
                'price_vs_fast': price_vs_fast,
                'rsi': rsi,
                'reason': 'Upward breakout with momentum'
            }
        }
    
    if price_vs_fast < -breakout_threshold and rsi < 40:
        return {
            'regime': 'breakout_down',
            'confidence': min(abs(price_vs_fast) / (breakout_threshold * 2) + (50 - rsi) / 50, 1.0),
            'metadata': {
                'price_vs_fast': price_vs_fast,
                'rsi': rsi,
                'reason': 'Downward breakout with momentum'
            }
        }
    
    if rsi < 25 and price_vs_fast < 0:
        return {
            'regime': 'reversal_up',
            'confidence': (25 - rsi) / 25,
            'metadata': {
                'rsi': rsi,
                'price_vs_fast': price_vs_fast,
                'reason': 'Oversold reversal setup'
            }
        }
    
    if rsi > 75 and price_vs_fast > 0:
        return {
            'regime': 'reversal_down',
            'confidence': (rsi - 75) / 25,
            'metadata': {
                'rsi': rsi,
                'price_vs_fast': price_vs_fast,
                'reason': 'Overbought reversal setup'
            }
        }
    
    return {
        'regime': 'consolidation',
        'confidence': 0.5,
        'metadata': {
            'reason': 'Default to consolidation'
        }
    }


@classifier(
    name='hidden_markov_classifier',
    regime_types=['accumulation', 'markup', 'distribution', 'markdown', 'uncertainty'],
    features=['volume', 'rsi_14', 'sma_20', 'sma_50', 'atr_14']
)
def hidden_markov_classifier(features: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Market phase classifier based on Wyckoff/HMM-inspired market cycles.
    
    States:
    - accumulation: Smart money accumulating, low volatility, sideways
    - markup: Uptrend phase, increasing volume and price
    - distribution: Smart money distributing, high volatility at tops
    - markdown: Downtrend phase, decreasing price
    - uncertainty: Transition phase, unclear direction
    """
    # Parameters
    volume_surge_threshold = params.get('volume_surge_threshold', 1.5)
    trend_strength_threshold = params.get('trend_strength_threshold', 0.02)
    volatility_threshold = params.get('volatility_threshold', 1.5)
    
    # Get features
    volume = features.get('volume', 0)
    rsi = features.get('rsi_14') or features.get('rsi')
    sma_20 = features.get('sma_20')
    sma_50 = features.get('sma_50')
    atr = features.get('atr_14') or features.get('atr')
    price = features.get('close', 0)
    
    # Get historical volume for comparison (simplified - would need rolling average in production)
    avg_volume = features.get('volume_sma_20', volume)  # Fallback to current if no average
    
    if not all([volume > 0, rsi, sma_20, sma_50, atr, price > 0]):
        return {
            'regime': 'uncertainty',
            'confidence': 0.0,
            'metadata': {'reason': 'Missing features'}
        }
    
    # Calculate key metrics
    volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
    trend_strength = (sma_20 - sma_50) / sma_50
    price_position = (price - sma_20) / sma_20
    volatility_pct = (atr / price) * 100
    
    # Market phase detection logic
    is_high_volume = volume_ratio > volume_surge_threshold
    is_low_volatility = volatility_pct < volatility_threshold
    is_high_volatility = volatility_pct > volatility_threshold * 1.5
    
    # Accumulation: Low volatility, sideways price, possible volume spikes
    if abs(trend_strength) < trend_strength_threshold and is_low_volatility:
        if is_high_volume and rsi < 50:
            regime = 'accumulation'
            confidence = min(volume_ratio / 2 + (50 - rsi) / 100, 1.0)
            reason = f'Low volatility sideways with volume surge, RSI: {rsi:.1f}'
        elif rsi < 40:
            regime = 'accumulation'
            confidence = (40 - rsi) / 40
            reason = f'Oversold sideways market, RSI: {rsi:.1f}'
        else:
            regime = 'uncertainty'
            confidence = 0.6
            reason = 'Sideways market, unclear accumulation'
    
    # Markup: Uptrend with increasing volume
    elif trend_strength > trend_strength_threshold and price_position > 0:
        if is_high_volume or rsi > 50:
            regime = 'markup'
            confidence = min(trend_strength / (trend_strength_threshold * 2) + volume_ratio / 3, 1.0)
            reason = f'Uptrend with {volume_ratio:.1f}x volume, RSI: {rsi:.1f}'
        else:
            regime = 'markup'
            confidence = trend_strength / (trend_strength_threshold * 2)
            reason = f'Uptrend, moderate volume'
    
    # Distribution: High volatility at tops, divergences
    elif is_high_volatility and rsi > 60 and price_position > 0.01:
        if is_high_volume:
            regime = 'distribution'
            confidence = min(volatility_pct / (volatility_threshold * 2) + (rsi - 60) / 40, 1.0)
            reason = f'High volatility top with {volume_ratio:.1f}x volume, RSI: {rsi:.1f}'
        else:
            regime = 'distribution'
            confidence = (rsi - 60) / 40 * volatility_pct / (volatility_threshold * 2)
            reason = f'Potential top, RSI: {rsi:.1f}, volatility: {volatility_pct:.2f}%'
    
    # Markdown: Downtrend phase
    elif trend_strength < -trend_strength_threshold and price_position < 0:
        regime = 'markdown'
        confidence = min(abs(trend_strength) / (trend_strength_threshold * 2) + (50 - rsi) / 50, 1.0)
        reason = f'Downtrend, RSI: {rsi:.1f}, trend: {trend_strength:.3f}'
    
    # Default: Uncertainty
    else:
        regime = 'uncertainty'
        confidence = 0.5
        reason = f'Mixed signals: trend={trend_strength:.3f}, vol={volatility_pct:.2f}%, RSI={rsi:.1f}'
    
    return {
        'regime': regime,
        'confidence': confidence,
        'metadata': {
            'volume_ratio': volume_ratio,
            'trend_strength': trend_strength,
            'price_position': price_position,
            'volatility_pct': volatility_pct,
            'rsi': rsi,
            'reason': reason
        }
    }