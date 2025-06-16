"""
Intraday Opening Range Breakout (ORB) Classifiers

Specialized classifiers for 1-minute timeframe trading that detect:
- Opening range breakouts (first 30-60 minutes)
- Intraday momentum shifts
- Session effects (market open, lunch, close)
- Microstructure patterns (volatility spikes, volume surges)
"""

from typing import Dict, Any, Optional
from datetime import datetime, time
from ...core.components.discovery import classifier


@classifier(
    name='intraday_orb_classifier',
    regime_types=['orb_breakout_up', 'orb_breakout_down', 'orb_range_bound', 
                  'session_open_vol', 'midday_drift', 'close_volatility'],
    feature_config=['high', 'low', 'close', 'volume', 'sma_5', 'atr_10']
)
def intraday_orb_classifier(features: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Six-state intraday classifier focusing on ORB and session effects.
    
    States:
    - orb_breakout_up: Price breaks above opening range with volume
    - orb_breakout_down: Price breaks below opening range with volume  
    - orb_range_bound: Price trading within opening range
    - session_open_vol: Opening session high volatility period
    - midday_drift: Low volatility midday consolidation
    - close_volatility: End-of-session volatility spike
    """
    # Extract session timing parameters
    orb_minutes = params.get('orb_minutes', 30)  # Opening range period
    open_session_minutes = params.get('open_session_minutes', 90)  # High vol period after open
    close_session_minutes = params.get('close_session_minutes', 60)  # High vol period before close
    
    # Breakout thresholds
    orb_breakout_threshold = params.get('orb_breakout_threshold', 0.002)  # 0.2%
    volume_surge_threshold = params.get('volume_surge_threshold', 1.5)
    volatility_threshold = params.get('volatility_threshold', 1.5)
    
    # Get current time (would come from market data timestamp in production)
    current_time = features.get('timestamp', datetime.now().time())
    if isinstance(current_time, datetime):
        current_time = current_time.time()
    
    # Get features
    high = features.get('high', 0)
    low = features.get('low', 0) 
    close = features.get('close', 0)
    volume = features.get('volume', 0)
    sma_5 = features.get('sma_5', close)
    atr = features.get('atr_10') or features.get('atr', 0)
    
    # ORB levels (would be calculated from first N minutes of session)
    orb_high = features.get('orb_high', high)  # Opening range high
    orb_low = features.get('orb_low', low)    # Opening range low
    orb_range = orb_high - orb_low if orb_high and orb_low else 0
    
    # Volume comparison (simplified - would use rolling average in production)
    avg_volume = features.get('volume_avg', volume)
    volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
    
    # Calculate key metrics
    if not all([high, low, close, orb_high, orb_low]) or orb_range == 0:
        return {
            'regime': 'orb_range_bound',
            'confidence': 0.0,
            'metadata': {'reason': 'Missing ORB levels or price data'}
        }
    
    # Position relative to ORB
    orb_breakout_up_pct = (close - orb_high) / orb_high if orb_high > 0 else 0
    orb_breakout_down_pct = (orb_low - close) / orb_low if orb_low > 0 else 0
    
    # Volatility level
    volatility_pct = (atr / close) * 100 if close > 0 and atr > 0 else 0
    
    # Session timing analysis
    market_open = time(9, 30)  # Adjust for your market
    market_close = time(16, 0)
    lunch_start = time(12, 0)
    lunch_end = time(13, 30)
    
    # Calculate minutes since market open (simplified)
    def minutes_since_open(current_time, market_open):
        if isinstance(current_time, time) and isinstance(market_open, time):
            current_minutes = current_time.hour * 60 + current_time.minute
            open_minutes = market_open.hour * 60 + market_open.minute
            return current_minutes - open_minutes
        return 60  # Default to post-opening
    
    minutes_from_open = minutes_since_open(current_time, market_open)
    minutes_to_close = 390 - minutes_from_open  # 6.5 hour session
    
    # Classification logic
    
    # 1. ORB Breakout Up: Above opening range with volume confirmation
    if orb_breakout_up_pct > orb_breakout_threshold:
        volume_conf = min(volume_ratio / volume_surge_threshold, 1.0) if volume_ratio > 1.0 else 0.5
        breakout_conf = min(orb_breakout_up_pct / (orb_breakout_threshold * 3), 1.0)
        
        return {
            'regime': 'orb_breakout_up',
            'confidence': (volume_conf + breakout_conf) / 2,
            'metadata': {
                'orb_breakout_pct': orb_breakout_up_pct * 100,
                'volume_ratio': volume_ratio,
                'orb_high': orb_high,
                'current_price': close,
                'minutes_from_open': minutes_from_open,
                'reason': f'Breakout above ORB high by {orb_breakout_up_pct*100:.2f}% with {volume_ratio:.1f}x volume'
            }
        }
    
    # 2. ORB Breakout Down: Below opening range with volume confirmation  
    if orb_breakout_down_pct > orb_breakout_threshold:
        volume_conf = min(volume_ratio / volume_surge_threshold, 1.0) if volume_ratio > 1.0 else 0.5
        breakout_conf = min(orb_breakout_down_pct / (orb_breakout_threshold * 3), 1.0)
        
        return {
            'regime': 'orb_breakout_down',
            'confidence': (volume_conf + breakout_conf) / 2,
            'metadata': {
                'orb_breakout_pct': orb_breakout_down_pct * 100,
                'volume_ratio': volume_ratio,
                'orb_low': orb_low,
                'current_price': close,
                'minutes_from_open': minutes_from_open,
                'reason': f'Breakout below ORB low by {orb_breakout_down_pct*100:.2f}% with {volume_ratio:.1f}x volume'
            }
        }
    
    # 3. Session Open Volatility: First 90 minutes with high volatility
    if minutes_from_open <= open_session_minutes and volatility_pct > volatility_threshold:
        vol_conf = min(volatility_pct / (volatility_threshold * 2), 1.0)
        time_conf = 1.0 - (minutes_from_open / open_session_minutes)
        
        return {
            'regime': 'session_open_vol',
            'confidence': (vol_conf + time_conf) / 2,
            'metadata': {
                'volatility_pct': volatility_pct,
                'minutes_from_open': minutes_from_open,
                'volume_ratio': volume_ratio,
                'reason': f'Opening session volatility: {volatility_pct:.2f}%, {minutes_from_open}min from open'
            }
        }
    
    # 4. Close Volatility: Last hour with high volatility or volume
    if minutes_to_close <= close_session_minutes and (volatility_pct > volatility_threshold or volume_ratio > volume_surge_threshold):
        vol_conf = min(volatility_pct / volatility_threshold, 1.0) if volatility_pct > 0 else 0
        volume_conf = min(volume_ratio / volume_surge_threshold, 1.0) if volume_ratio > 1 else 0
        time_conf = 1.0 - (minutes_to_close / close_session_minutes)
        
        return {
            'regime': 'close_volatility', 
            'confidence': max(vol_conf, volume_conf) * time_conf,
            'metadata': {
                'volatility_pct': volatility_pct,
                'volume_ratio': volume_ratio,
                'minutes_to_close': minutes_to_close,
                'reason': f'Close session activity: vol {volatility_pct:.2f}%, volume {volume_ratio:.1f}x, {minutes_to_close}min to close'
            }
        }
    
    # 5. Midday Drift: Low volatility during lunch/midday hours
    if (lunch_start <= current_time <= lunch_end or 
        (minutes_from_open > 150 and minutes_to_close > 90)) and volatility_pct < volatility_threshold * 0.5:
        
        drift_conf = 1.0 - (volatility_pct / volatility_threshold)
        
        return {
            'regime': 'midday_drift',
            'confidence': drift_conf,
            'metadata': {
                'volatility_pct': volatility_pct,
                'current_time': str(current_time),
                'minutes_from_open': minutes_from_open,
                'minutes_to_close': minutes_to_close,
                'reason': f'Low volatility midday: {volatility_pct:.2f}%'
            }
        }
    
    # 6. Default: ORB Range Bound - trading within opening range
    orb_position = (close - orb_low) / orb_range if orb_range > 0 else 0.5
    range_conf = 1.0 - abs(orb_position - 0.5) * 2  # Higher confidence when near middle
    
    return {
        'regime': 'orb_range_bound',
        'confidence': range_conf,
        'metadata': {
            'orb_position': orb_position,
            'orb_range': orb_range,
            'orb_high': orb_high,
            'orb_low': orb_low,
            'current_price': close,
            'volatility_pct': volatility_pct,
            'volume_ratio': volume_ratio,
            'minutes_from_open': minutes_from_open,
            'reason': f'Trading within ORB range, position {orb_position:.2f}'
        }
    }


@classifier(
    name='microstructure_momentum_classifier',
    regime_types=['momentum_acceleration', 'momentum_deceleration', 'volume_breakout', 'liquidity_void', 'normal_flow'],
    feature_config=['close', 'volume', 'sma_3', 'sma_5', 'atr_5']
)
def microstructure_momentum_classifier(features: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Five-state microstructure momentum classifier for 1m trading.
    
    Focuses on very short-term momentum and order flow patterns:
    - momentum_acceleration: Price moving with increasing momentum
    - momentum_deceleration: Price momentum slowing down
    - volume_breakout: Sudden volume spike with price movement
    - liquidity_void: Low volume, potential for quick moves
    - normal_flow: Regular trading flow
    """
    # Parameters
    momentum_accel_threshold = params.get('momentum_accel_threshold', 0.001)  # 0.1%
    volume_spike_threshold = params.get('volume_spike_threshold', 2.0)
    liquidity_threshold = params.get('liquidity_threshold', 0.5)  # Low volume threshold
    
    # Get features
    close = features.get('close', 0)
    volume = features.get('volume', 0)
    sma_3 = features.get('sma_3', close)
    sma_5 = features.get('sma_5', close)
    atr_5 = features.get('atr_5') or features.get('atr', 0)
    
    # Previous values (would come from lookback in production)
    prev_close = features.get('prev_close', close)
    prev_volume = features.get('prev_volume', volume)
    avg_volume = features.get('volume_avg', volume)
    
    if not all([close > 0, volume > 0, sma_3 > 0, sma_5 > 0]):
        return {
            'regime': 'normal_flow',
            'confidence': 0.0,
            'metadata': {'reason': 'Missing price/volume data'}
        }
    
    # Calculate momentum metrics
    price_momentum = (close - prev_close) / prev_close if prev_close > 0 else 0
    sma_momentum = (sma_3 - sma_5) / sma_5 if sma_5 > 0 else 0
    volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
    volume_acceleration = (volume - prev_volume) / prev_volume if prev_volume > 0 else 0
    
    # Momentum acceleration detection
    momentum_strength = abs(price_momentum)
    momentum_direction = 1 if price_momentum > 0 else -1
    
    # 1. Momentum Acceleration: Strong price movement with increasing momentum
    if momentum_strength > momentum_accel_threshold and abs(sma_momentum) > momentum_accel_threshold * 0.5:
        accel_conf = min(momentum_strength / (momentum_accel_threshold * 3), 1.0)
        trend_conf = min(abs(sma_momentum) / momentum_accel_threshold, 1.0)
        
        return {
            'regime': 'momentum_acceleration',
            'confidence': (accel_conf + trend_conf) / 2,
            'metadata': {
                'price_momentum': price_momentum * 100,
                'sma_momentum': sma_momentum * 100,
                'direction': 'bullish' if momentum_direction > 0 else 'bearish',
                'volume_ratio': volume_ratio,
                'reason': f'Accelerating {("bullish" if momentum_direction > 0 else "bearish")} momentum: {momentum_strength*100:.3f}%'
            }
        }
    
    # 2. Volume Breakout: Sudden volume spike with price movement
    if volume_ratio > volume_spike_threshold and momentum_strength > momentum_accel_threshold * 0.5:
        volume_conf = min(volume_ratio / (volume_spike_threshold * 1.5), 1.0)
        price_conf = momentum_strength / momentum_accel_threshold
        
        return {
            'regime': 'volume_breakout',
            'confidence': min((volume_conf + price_conf) / 2, 1.0),
            'metadata': {
                'volume_ratio': volume_ratio,
                'price_momentum': price_momentum * 100,
                'volume_acceleration': volume_acceleration * 100,
                'direction': 'bullish' if momentum_direction > 0 else 'bearish',
                'reason': f'Volume breakout: {volume_ratio:.1f}x avg with {momentum_strength*100:.3f}% move'
            }
        }
    
    # 3. Liquidity Void: Very low volume, potential for quick moves
    if volume_ratio < liquidity_threshold:
        liquidity_conf = 1.0 - (volume_ratio / liquidity_threshold)
        
        return {
            'regime': 'liquidity_void',
            'confidence': liquidity_conf,
            'metadata': {
                'volume_ratio': volume_ratio,
                'avg_volume': avg_volume,
                'current_volume': volume,
                'price_momentum': price_momentum * 100,
                'reason': f'Low liquidity: {volume_ratio:.2f}x avg volume'
            }
        }
    
    # 4. Momentum Deceleration: Slowing momentum after previous moves
    if momentum_strength < momentum_accel_threshold * 0.5 and abs(sma_momentum) < momentum_accel_threshold * 0.3:
        decel_conf = 1.0 - (momentum_strength / momentum_accel_threshold)
        
        return {
            'regime': 'momentum_deceleration',
            'confidence': decel_conf,
            'metadata': {
                'price_momentum': price_momentum * 100,
                'sma_momentum': sma_momentum * 100,
                'volume_ratio': volume_ratio,
                'reason': f'Momentum deceleration: {momentum_strength*100:.3f}% move, vol {volume_ratio:.1f}x'
            }
        }
    
    # 5. Default: Normal Flow
    return {
        'regime': 'normal_flow',
        'confidence': 0.6,
        'metadata': {
            'price_momentum': price_momentum * 100,
            'volume_ratio': volume_ratio,
            'sma_momentum': sma_momentum * 100,
            'reason': f'Normal flow: {momentum_strength*100:.3f}% move, {volume_ratio:.1f}x volume'
        }
    }


@classifier(
    name='session_pattern_classifier', 
    regime_types=['gap_up', 'gap_down', 'opening_auction', 'trending_session', 'consolidation_session'],
    feature_config=['open', 'high', 'low', 'close', 'volume']
)
def session_pattern_classifier(features: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Five-state session pattern classifier for intraday patterns.
    
    Detects daily session characteristics:
    - gap_up: Significant gap up from previous close
    - gap_down: Significant gap down from previous close  
    - opening_auction: High volume opening phase
    - trending_session: Sustained directional movement
    - consolidation_session: Range-bound session
    """
    # Parameters
    gap_threshold = params.get('gap_threshold', 0.005)  # 0.5%
    trend_session_threshold = params.get('trend_session_threshold', 0.01)  # 1%
    opening_volume_threshold = params.get('opening_volume_threshold', 2.0)
    
    # Get features
    open_price = features.get('open', 0)
    high = features.get('high', 0)
    low = features.get('low', 0)
    close = features.get('close', 0)
    volume = features.get('volume', 0)
    
    # Previous session data
    prev_close = features.get('prev_close', open_price)
    avg_volume = features.get('avg_volume', volume)
    session_high = features.get('session_high', high)
    session_low = features.get('session_low', low)
    
    if not all([open_price > 0, high > 0, low > 0, close > 0, prev_close > 0]):
        return {
            'regime': 'consolidation_session',
            'confidence': 0.0,
            'metadata': {'reason': 'Missing session data'}
        }
    
    # Calculate session metrics
    gap_pct = (open_price - prev_close) / prev_close
    session_range_pct = (session_high - session_low) / session_low if session_low > 0 else 0
    session_direction = (close - open_price) / open_price if open_price > 0 else 0
    volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
    
    # 1. Gap Up: Significant gap higher
    if gap_pct > gap_threshold:
        gap_conf = min(gap_pct / (gap_threshold * 3), 1.0)
        
        return {
            'regime': 'gap_up',
            'confidence': gap_conf,
            'metadata': {
                'gap_pct': gap_pct * 100,
                'prev_close': prev_close,
                'open_price': open_price,
                'volume_ratio': volume_ratio,
                'reason': f'Gap up {gap_pct*100:.2f}% from previous close'
            }
        }
    
    # 2. Gap Down: Significant gap lower  
    if gap_pct < -gap_threshold:
        gap_conf = min(abs(gap_pct) / (gap_threshold * 3), 1.0)
        
        return {
            'regime': 'gap_down',
            'confidence': gap_conf,
            'metadata': {
                'gap_pct': gap_pct * 100,
                'prev_close': prev_close,
                'open_price': open_price,
                'volume_ratio': volume_ratio,
                'reason': f'Gap down {abs(gap_pct)*100:.2f}% from previous close'
            }
        }
    
    # 3. Opening Auction: High volume opening phase
    if volume_ratio > opening_volume_threshold and abs(gap_pct) > gap_threshold * 0.5:
        volume_conf = min(volume_ratio / (opening_volume_threshold * 1.5), 1.0)
        gap_conf = abs(gap_pct) / gap_threshold
        
        return {
            'regime': 'opening_auction',
            'confidence': (volume_conf + gap_conf) / 2,
            'metadata': {
                'volume_ratio': volume_ratio,
                'gap_pct': gap_pct * 100,
                'session_range_pct': session_range_pct * 100,
                'reason': f'Opening auction: {volume_ratio:.1f}x volume, {abs(gap_pct)*100:.2f}% gap'
            }
        }
    
    # 4. Trending Session: Sustained directional movement
    if abs(session_direction) > trend_session_threshold and session_range_pct > trend_session_threshold:
        trend_conf = min(abs(session_direction) / (trend_session_threshold * 2), 1.0)
        range_conf = min(session_range_pct / (trend_session_threshold * 3), 1.0)
        
        return {
            'regime': 'trending_session',
            'confidence': (trend_conf + range_conf) / 2,
            'metadata': {
                'session_direction': session_direction * 100,
                'session_range_pct': session_range_pct * 100,
                'direction': 'bullish' if session_direction > 0 else 'bearish',
                'volume_ratio': volume_ratio,
                'reason': f'Trending session: {abs(session_direction)*100:.2f}% {"up" if session_direction > 0 else "down"}'
            }
        }
    
    # 5. Default: Consolidation Session
    return {
        'regime': 'consolidation_session',
        'confidence': 0.7,
        'metadata': {
            'session_direction': session_direction * 100,
            'session_range_pct': session_range_pct * 100,
            'gap_pct': gap_pct * 100,
            'volume_ratio': volume_ratio,
            'reason': f'Consolidation: {session_range_pct*100:.2f}% range, {session_direction*100:.2f}% direction'
        }
    }