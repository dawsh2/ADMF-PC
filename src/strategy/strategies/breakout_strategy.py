"""
Breakout Strategy

Trades breakouts from recent price ranges with volume confirmation.
"""

import logging
from typing import Dict, Any, Optional

from ...core.components.discovery import strategy

logger = logging.getLogger(__name__)


@strategy(
    name='breakout_strategy',
    feature_config={
        'high': {'params': ['lookback_period'], 'default': 20},
        'low': {'params': ['lookback_period'], 'default': 20},
        'volume': {'params': ['lookback_period'], 'default': 20},
        'atr': {'params': ['atr_period'], 'default': 14}
    }
)
def breakout_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Breakout trading strategy.
    
    Entry:
    - Long when price breaks above recent high with volume
    - Short when price breaks below recent low with volume
    
    Exit:
    - Stop loss based on ATR
    - Take profit at 2x risk
    """
    # Parameters
    lookback_period = params.get('lookback_period', 20)
    breakout_mult = params.get('breakout_mult', 1.0)  # Multiplier for breakout threshold
    volume_mult = params.get('volume_mult', 1.5)     # Volume must be this x average
    stop_loss_atr = params.get('stop_loss_atr', 2.0) # Stop loss in ATR units
    
    # Get features
    recent_high = features.get(f'high_{lookback_period}', features.get('recent_high'))
    recent_low = features.get(f'low_{lookback_period}', features.get('recent_low'))
    current_volume = bar.get('volume', features.get('volume'))
    avg_volume = features.get(f'volume_{lookback_period}_volume_ma', features.get('avg_volume'))
    atr = features.get('atr', features.get('atr_14'))
    
    # Get current price and symbol
    current_price = bar.get('close', 0)
    symbol = bar.get('symbol', 'UNKNOWN')
    
    if recent_high is None or recent_low is None or current_volume is None or avg_volume is None:
        return None
    
    # Calculate breakout thresholds
    range_size = recent_high - recent_low
    if range_size <= 0:
        return None
        
    upper_threshold = recent_high + (range_size * (breakout_mult - 1))
    lower_threshold = recent_low - (range_size * (breakout_mult - 1))
    
    # Check volume condition
    volume_confirmed = current_volume > (avg_volume * volume_mult) if avg_volume > 0 else False
    
    # Generate signals
    if current_price > upper_threshold and volume_confirmed:
        signal = {
            'symbol': symbol,
            'direction': 'long',
            'signal_type': 'entry',
            'strength': min((current_price - upper_threshold) / range_size, 1.0),
            'price': current_price,
            'reason': f'Bullish breakout: price {current_price:.2f} > {upper_threshold:.2f}, volume {current_volume/avg_volume:.1f}x avg',
            'indicators': {
                'recent_high': recent_high,
                'recent_low': recent_low,
                'upper_threshold': upper_threshold,
                'volume_ratio': current_volume/avg_volume if avg_volume > 0 else 0,
                'atr': atr
            },
            'stop_loss': current_price - (atr * stop_loss_atr) if atr else None,
            'take_profit': current_price + (atr * stop_loss_atr * 2) if atr else None
        }
    elif current_price < lower_threshold and volume_confirmed:
        signal = {
            'symbol': symbol,
            'direction': 'short',
            'signal_type': 'entry',
            'strength': min((lower_threshold - current_price) / range_size, 1.0),
            'price': current_price,
            'reason': f'Bearish breakout: price {current_price:.2f} < {lower_threshold:.2f}, volume {current_volume/avg_volume:.1f}x avg',
            'indicators': {
                'recent_high': recent_high,
                'recent_low': recent_low,
                'lower_threshold': lower_threshold,
                'volume_ratio': current_volume/avg_volume if avg_volume > 0 else 0,
                'atr': atr
            },
            'stop_loss': current_price + (atr * stop_loss_atr) if atr else None,
            'take_profit': current_price - (atr * stop_loss_atr * 2) if atr else None
        }
    else:
        signal = {
            'symbol': symbol,
            'direction': 'flat',
            'signal_type': 'entry',
            'strength': 0.0,
            'price': current_price,
            'reason': f'No breakout: price within range [{lower_threshold:.2f}, {upper_threshold:.2f}]',
            'indicators': {
                'recent_high': recent_high,
                'recent_low': recent_low,
                'current_price': current_price,
                'volume_confirmed': volume_confirmed
            }
        }
    
    return signal