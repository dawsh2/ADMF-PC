"""
Enhanced Multi-State Classifiers

These classifiers implement more granular regime classification with 3-5 states
for better regime-aware strategy analysis.
"""

from typing import Dict, Any
from ...core.components.discovery import classifier


@classifier(
    name='multi_timeframe_trend_classifier',
    regime_types=['strong_uptrend', 'weak_uptrend', 'sideways', 'weak_downtrend', 'strong_downtrend'],
    feature_config=['sma', 'close'],
    param_feature_mapping={
        'sma_short': 'sma_{sma_short}',
        'sma_medium': 'sma_{sma_medium}',
        'sma_long': 'sma_{sma_long}'
    }
)
def multi_timeframe_trend_classifier(features: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Five-state trend classifier using multiple timeframe analysis.
    
    States:
    - strong_uptrend: All MAs aligned upward, strong momentum
    - weak_uptrend: MAs mostly upward, weak momentum  
    - sideways: MAs converging, no clear direction
    - weak_downtrend: MAs mostly downward, weak momentum
    - strong_downtrend: All MAs aligned downward, strong momentum
    """
    # Parameters - adjusted for better balance
    strong_threshold = params.get('strong_threshold', 0.01)  # 1% - reduced from 2%
    weak_threshold = params.get('weak_threshold', 0.002)    # 0.2% - reduced from 0.5%
    
    # Get periods from parameters with defaults
    sma_short_period = params.get('sma_short', 10)
    sma_medium_period = params.get('sma_medium', 20)
    sma_long_period = params.get('sma_long', 50)
    
    # Get moving averages using parameterized names
    sma_10 = features.get(f'sma_{sma_short_period}')
    sma_20 = features.get(f'sma_{sma_medium_period}') 
    sma_50 = features.get(f'sma_{sma_long_period}')
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
    
    # Weighted average trend strength - price trend gets more weight
    avg_trend = (short_trend * 0.2 + medium_trend * 0.3 + price_trend * 0.5)
    
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
    feature_config=['atr', 'rsi', 'sma', 'close'],
    param_feature_mapping={
        'atr_period': 'atr_{atr_period}',
        'rsi_period': 'rsi_{rsi_period}',
        'sma_period': 'sma_{sma_period}'
    }
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
    # Parameters - adjusted for better balance
    vol_threshold = params.get('vol_threshold', 1.0)      # 1% - reduced from 1.5%
    rsi_overbought = params.get('rsi_overbought', 60)    # 60 - reduced from 65
    rsi_oversold = params.get('rsi_oversold', 40)        # 40 - increased from 35
    
    # Get periods from parameters with defaults
    atr_period = params.get('atr_period', 14)
    rsi_period = params.get('rsi_period', 14)
    sma_period = params.get('sma_period', 20)
    
    # Get features using parameterized names
    atr = features.get(f'atr_{atr_period}')
    rsi = features.get(f'rsi_{rsi_period}')
    sma = features.get(f'sma_{sma_period}')
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
    
    # Determine momentum direction - use OR logic for better balance
    price_momentum = (price - sma) / sma
    is_bullish = rsi > rsi_overbought or (rsi > 55 and price_momentum > 0.005)
    is_bearish = rsi < rsi_oversold or (rsi < 45 and price_momentum < -0.005)
    
    # Classification
    if is_high_vol:
        if is_bullish:
            regime = 'high_vol_bullish'
            confidence = min((rsi - rsi_overbought) / (100 - rsi_overbought) + vol_pct / (vol_threshold * 2), 1.0)
        elif is_bearish:
            regime = 'high_vol_bearish' 
            confidence = min((rsi_oversold - rsi) / rsi_oversold + vol_pct / (vol_threshold * 2), 1.0)
        else:
            regime = 'neutral'
            confidence = 0.5
    else:
        if is_bullish:
            regime = 'low_vol_bullish'
            confidence = (rsi - rsi_overbought) / (100 - rsi_overbought) * (1 - vol_pct / vol_threshold)
        elif is_bearish:
            regime = 'low_vol_bearish'
            confidence = (rsi_oversold - rsi) / rsi_oversold * (1 - vol_pct / vol_threshold)
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
    feature_config=['sma', 'atr', 'rsi', 'close'],
    param_feature_mapping={
        'sma_short': 'sma_{sma_short}',
        'sma_long': 'sma_{sma_long}',
        'atr_period': 'atr_{atr_period}',
        'rsi_period': 'rsi_{rsi_period}'
    }
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
    # Parameters - adjusted for better balance
    trend_threshold = params.get('trend_threshold', 0.005)  # 0.5% - reduced from 1%
    vol_threshold = params.get('vol_threshold', 0.8)       # 0.8% - reduced from 1%
    
    # Get periods from parameters with defaults
    sma_short_period = params.get('sma_short', 10)
    sma_long_period = params.get('sma_long', 50)
    atr_period = params.get('atr_period', 20)
    rsi_period = params.get('rsi_period', 14)
    
    # Get features using parameterized names
    sma_short = features.get(f'sma_{sma_short_period}')
    sma_long = features.get(f'sma_{sma_long_period}')
    atr = features.get(f'atr_{atr_period}')
    rsi = features.get(f'rsi_{rsi_period}')
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
    
    # Determine market bias - improved logic
    is_bullish = trend_strength > 0 and rsi > 48  # More lenient RSI
    is_bearish = trend_strength < 0 and rsi < 52  # More lenient RSI
    is_trending = abs(trend_strength) > trend_threshold or vol_level > vol_threshold  # OR not AND
    
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
    feature_config=['sma', 'atr', 'rsi', 'close'],
    param_feature_mapping={
        'sma_fast': 'sma_{sma_fast}',
        'sma_slow': 'sma_{sma_slow}',
        'atr_period': 'atr_{atr_period}',
        'rsi_period': 'rsi_{rsi_period}'
    }
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
    # Parameters - adjusted for better balance
    breakout_threshold = params.get('breakout_threshold', 0.003)  # 0.3% - reduced from 0.5%
    consolidation_threshold = params.get('consolidation_threshold', 0.001)  # 0.1% - reduced from 0.2%
    
    # Get periods from parameters with defaults
    sma_fast_period = params.get('sma_fast', 5)
    sma_slow_period = params.get('sma_slow', 20)
    atr_period = params.get('atr_period', 10)
    rsi_period = params.get('rsi_period', 7)
    
    # Get features using parameterized names
    sma_fast = features.get(f'sma_{sma_fast_period}')
    sma_slow = features.get(f'sma_{sma_slow_period}')
    atr = features.get(f'atr_{atr_period}')
    rsi = features.get(f'rsi_{rsi_period}')
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
    
    if rsi < 30 and price_vs_fast < 0:  # More lenient RSI threshold
        return {
            'regime': 'reversal_up',
            'confidence': (30 - rsi) / 30,
            'metadata': {
                'rsi': rsi,
                'price_vs_fast': price_vs_fast,
                'reason': 'Oversold reversal setup'
            }
        }
    
    if rsi > 70 and price_vs_fast > 0:  # More lenient RSI threshold
        return {
            'regime': 'reversal_down',
            'confidence': (rsi - 70) / 30,
            'metadata': {
                'rsi': rsi,
                'price_vs_fast': price_vs_fast,
                'reason': 'Overbought reversal setup'
            }
        }
    
    # More nuanced default classification based on metrics
    if vol_pct < 0.8:  # Low volatility
        return {
            'regime': 'consolidation',
            'confidence': 0.7,
            'metadata': {
                'vol_pct': vol_pct,
                'reason': 'Low volatility consolidation'
            }
        }
    elif price_vs_fast > 0 and rsi > 50:
        return {
            'regime': 'breakout_up',
            'confidence': 0.4,
            'metadata': {
                'price_vs_fast': price_vs_fast,
                'rsi': rsi,
                'reason': 'Mild upward pressure'
            }
        }
    elif price_vs_fast < 0 and rsi < 50:
        return {
            'regime': 'breakout_down',
            'confidence': 0.4,
            'metadata': {
                'price_vs_fast': price_vs_fast,
                'rsi': rsi,
                'reason': 'Mild downward pressure'
            }
        }
    else:
        return {
            'regime': 'consolidation',
            'confidence': 0.5,
            'metadata': {
                'reason': 'Mixed signals'
            }
        }


@classifier(
    name='hidden_markov_classifier',
    regime_types=['accumulation', 'markup', 'distribution', 'markdown', 'uncertainty'],
    feature_config=['volume', 'rsi', 'sma', 'atr', 'close'],
    param_feature_mapping={
        'rsi_period': 'rsi_{rsi_period}',
        'sma_short': 'sma_{sma_short}',
        'sma_long': 'sma_{sma_long}',
        'atr_period': 'atr_{atr_period}'
    }
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
    # Parameters - adjusted for better balance
    volume_surge_threshold = params.get('volume_surge_threshold', 1.3)      # 1.3x - reduced from 1.5x
    trend_strength_threshold = params.get('trend_strength_threshold', 0.01) # 1% - reduced from 2%
    volatility_threshold = params.get('volatility_threshold', 1.2)         # 1.2% - reduced from 1.5%
    
    # Get periods from parameters with defaults
    rsi_period = params.get('rsi_period', 14)
    sma_short_period = params.get('sma_short', 20)
    sma_long_period = params.get('sma_long', 50)
    atr_period = params.get('atr_period', 14)
    
    # Get features using parameterized names
    volume = features.get('volume', 0)
    rsi = features.get(f'rsi_{rsi_period}')
    sma_20 = features.get(f'sma_{sma_short_period}')
    sma_50 = features.get(f'sma_{sma_long_period}')
    atr = features.get(f'atr_{atr_period}')
    price = features.get('close', 0)
    
    # Get historical volume for comparison (simplified - would need rolling average in production)
    avg_volume = features.get(f'volume_sma_{sma_short_period}', volume)  # Fallback to current if no average
    
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
    is_high_volatility = volatility_pct > volatility_threshold * 1.3  # Less strict multiplier
    
    # Accumulation: Low volatility, sideways price, possible volume spikes
    if abs(trend_strength) < trend_strength_threshold and is_low_volatility:
        if is_high_volume or rsi < 45:  # More lenient conditions
            regime = 'accumulation'
            confidence = min(volume_ratio / 2 + (50 - rsi) / 100, 1.0)
            reason = f'Low volatility sideways, RSI: {rsi:.1f}, Vol: {volume_ratio:.1f}x'
        else:
            regime = 'accumulation'  # Default to accumulation not uncertainty
            confidence = 0.5
            reason = 'Sideways consolidation phase'
    
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
    elif (is_high_volatility or rsi > 65) and price_position > 0:  # More lenient
        regime = 'distribution'
        confidence = min(volatility_pct / (volatility_threshold * 2) + max(0, rsi - 60) / 40, 1.0)
        reason = f'Distribution phase, RSI: {rsi:.1f}, Vol: {volatility_pct:.2f}%'
    
    # Markdown: Downtrend phase
    elif trend_strength < -trend_strength_threshold and price_position < 0:
        regime = 'markdown'
        confidence = min(abs(trend_strength) / (trend_strength_threshold * 2) + (50 - rsi) / 50, 1.0)
        reason = f'Downtrend, RSI: {rsi:.1f}, trend: {trend_strength:.3f}'
    
    # Default: Assign based on dominant signal
    else:
        # Determine most likely regime based on individual signals
        if rsi > 55 and trend_strength > 0:
            regime = 'markup'
            confidence = 0.4
            reason = f'Weak bullish bias: trend={trend_strength:.3f}, RSI={rsi:.1f}'
        elif rsi < 45 and trend_strength < 0:
            regime = 'markdown'
            confidence = 0.4
            reason = f'Weak bearish bias: trend={trend_strength:.3f}, RSI={rsi:.1f}'
        elif volatility_pct > volatility_threshold:
            regime = 'distribution' if rsi > 50 else 'accumulation'
            confidence = 0.4
            reason = f'High volatility: {volatility_pct:.2f}%, RSI={rsi:.1f}'
        else:
            regime = 'accumulation'  # Default to accumulation over uncertainty
            confidence = 0.3
            reason = f'Low conviction signals: trend={trend_strength:.3f}, vol={volatility_pct:.2f}%'
    
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