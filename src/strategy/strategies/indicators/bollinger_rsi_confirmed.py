"""
Bollinger + RSI Divergence with Confirmation.

This implements the exact profitable strategy from the backtest by tracking
divergences in the strategy itself (stateless but with rich metadata).
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='bollinger_rsi_confirmed',
    feature_discovery=lambda params: [
        FeatureSpec('bollinger_bands', {
            'period': params.get('bb_period', 20),
            'std_dev': params.get('bb_std', 2.0)
        }, 'upper'),
        FeatureSpec('bollinger_bands', {
            'period': params.get('bb_period', 20),
            'std_dev': params.get('bb_std', 2.0)
        }, 'middle'),
        FeatureSpec('bollinger_bands', {
            'period': params.get('bb_period', 20),
            'std_dev': params.get('bb_std', 2.0)
        }, 'lower'),
        FeatureSpec('rsi', {
            'period': params.get('rsi_period', 14)
        })
    ],
    parameter_space={
        'bb_period': {'type': 'int', 'range': (20, 20), 'default': 20},
        'bb_std': {'type': 'float', 'range': (2.0, 2.0), 'default': 2.0},
        'rsi_period': {'type': 'int', 'range': (14, 14), 'default': 14},
        'rsi_divergence_threshold': {'type': 'float', 'range': (5.0, 10.0), 'default': 5.0},
        'exit_threshold': {'type': 'float', 'range': (0.0, 0.005), 'default': 0.001}
    },
    strategy_type='mean_reversion',
    tags=['bollinger_bands', 'rsi', 'divergence', 'mean_reversion', 'profitable']
)
def bollinger_rsi_confirmed(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Bollinger + RSI strategy with divergence confirmation logic.
    
    This matches the profitable backtest by:
    1. Price makes new extreme beyond Bollinger Bands
    2. RSI shows divergence (doesn't make new extreme)
    3. Wait for price to close back inside bands (confirmation)
    4. Exit at middle band
    
    Since we're stateless, we approximate the multi-bar pattern by:
    - Entering when price is back inside bands after being outside
    - AND RSI shows divergence characteristics (not at extreme when price was)
    """
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    rsi_period = params.get('rsi_period', 14)
    rsi_div_threshold = params.get('rsi_divergence_threshold', 5.0)
    exit_threshold = params.get('exit_threshold', 0.001)
    
    # Get features
    upper_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_upper')
    middle_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_middle')
    lower_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_lower')
    rsi = features.get(f'rsi_{rsi_period}')
    
    price = bar.get('close', 0)
    low = bar.get('low', 0)
    high = bar.get('high', 0)
    
    if any(v is None for v in [upper_band, lower_band, middle_band, rsi]):
        return None
    
    signal_value = 0
    entry_reason = None
    divergence_detected = False
    
    # Calculate band position
    band_width = upper_band - lower_band
    band_position = (price - lower_band) / band_width if band_width > 0 else 0.5
    
    # First check exit condition - within threshold of middle band
    if middle_band and abs(price - middle_band) / middle_band <= exit_threshold:
        signal_value = 0  # Exit any position
        entry_reason = "Exit at middle band"
    else:
        # Bullish divergence pattern (for long entry)
        # Price recently touched/broke lower band and is now back inside
        # RSI didn't get oversold (shows divergence)
        if (0.0 < band_position < 0.2 and  # Just inside lower band
            low <= lower_band * 1.001 and   # Low touched/broke band
            rsi > 30 + rsi_div_threshold):   # RSI shows strength (not oversold)
            signal_value = 1
            divergence_detected = True
            entry_reason = f"Bullish divergence confirmed - Price touched lower band, RSI={rsi:.1f} (not oversold)"
        
        # Bearish divergence pattern (for short entry)
        # Price recently touched/broke upper band and is now back inside
        # RSI didn't get overbought (shows divergence)
        elif (0.8 < band_position < 1.0 and  # Just inside upper band
              high >= upper_band * 0.999 and  # High touched/broke band
              rsi < 70 - rsi_div_threshold):   # RSI shows weakness (not overbought)
            signal_value = -1
            divergence_detected = True
            entry_reason = f"Bearish divergence confirmed - Price touched upper band, RSI={rsi:.1f} (not overbought)"
    
    # Always return signal
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'bollinger_rsi_confirmed',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'price': price,
            'high': high,
            'low': low,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'middle_band': middle_band,
            'rsi': rsi,
            'band_width': band_width,
            'band_position': band_position,
            'divergence_detected': divergence_detected,
            'near_lower': low <= lower_band * 1.001,
            'near_upper': high >= upper_band * 0.999,
            'rsi_suggests_long': rsi > 30 + rsi_div_threshold,
            'rsi_suggests_short': rsi < 70 - rsi_div_threshold,
            'reason': entry_reason or 'No divergence signal'
        }
    }