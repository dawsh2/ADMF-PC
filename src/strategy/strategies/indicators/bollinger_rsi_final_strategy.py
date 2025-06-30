"""
Final Bollinger + RSI Divergence Strategy.

This uses the bb_rsi_divergence_proper feature which tracks
the exact multi-bar pattern from the profitable backtest.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='bollinger_rsi_final',
    feature_discovery=lambda params: [
        FeatureSpec('bb_rsi_divergence_proper', {
            'bb_period': params.get('bb_period', 20),
            'bb_std': params.get('bb_std', 2.0),
            'rsi_period': params.get('rsi_period', 14),
            'lookback': params.get('lookback_bars', 20),
            'rsi_divergence_threshold': params.get('rsi_divergence_threshold', 5.0),
            'confirmation_bars': params.get('confirmation_bars', 10)
        })
    ],
    parameter_space={
        'bb_period': {'type': 'int', 'range': (20, 20), 'default': 20},
        'bb_std': {'type': 'float', 'range': (2.0, 2.0), 'default': 2.0},
        'rsi_period': {'type': 'int', 'range': (14, 14), 'default': 14},
        'rsi_divergence_threshold': {'type': 'float', 'range': (5.0, 10.0), 'default': 5.0},
        'lookback_bars': {'type': 'int', 'range': (20, 20), 'default': 20},
        'confirmation_bars': {'type': 'int', 'range': (10, 10), 'default': 10},
        'exit_threshold': {'type': 'float', 'range': (0.0, 0.005), 'default': 0.001}
    },
    strategy_type='mean_reversion',
    tags=['bollinger_bands', 'rsi', 'divergence', 'mean_reversion', 'final']
)
def bollinger_rsi_final(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Final implementation using the proper divergence tracking feature.
    
    The bb_rsi_divergence_proper feature handles all the complexity:
    - Computing Bollinger Bands and RSI
    - Tracking extremes when price goes outside bands
    - Detecting multi-bar divergence patterns
    - Confirming when price returns inside bands
    
    This strategy just reads the signals and manages exits.
    """
    exit_threshold = params.get('exit_threshold', 0.001)
    
    # Get the divergence feature data - it contains everything we need
    div_data = features.get('bb_rsi_divergence_proper', {})
    
    # Extract values from the feature
    upper_band = div_data.get('upper_band')
    middle_band = div_data.get('middle_band')
    lower_band = div_data.get('lower_band')
    rsi = div_data.get('rsi')
    
    price = bar.get('close', 0)
    
    if any(v is None for v in [upper_band, lower_band, middle_band]):
        return None
    
    signal_value = 0
    entry_reason = None
    
    # First check exit condition - within threshold of middle band
    if middle_band and abs(price - middle_band) / middle_band <= exit_threshold:
        signal_value = 0  # Exit any position
        entry_reason = "Exit at middle band"
    else:
        # Check for confirmed divergence signals from our feature
        if div_data.get('confirmed_long', False):
            signal_value = 1  # Long entry
            entry_reason = f"Bullish divergence confirmed - Strength: {div_data.get('divergence_strength', 0):.1f}, Bars since: {div_data.get('bars_since_divergence', 0)}"
        elif div_data.get('confirmed_short', False):
            signal_value = -1  # Short entry
            entry_reason = f"Bearish divergence confirmed - Strength: {div_data.get('divergence_strength', 0):.1f}, Bars since: {div_data.get('bars_since_divergence', 0)}"
    
    # Always return signal
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'bollinger_rsi_final',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'price': price,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'middle_band': middle_band,
            'rsi': rsi,
            'band_width': upper_band - lower_band if upper_band and lower_band else 0,
            'band_position': (price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5,
            'has_bullish_divergence': div_data.get('has_bullish_divergence', False),
            'has_bearish_divergence': div_data.get('has_bearish_divergence', False),
            'confirmed_long': div_data.get('confirmed_long', False),
            'confirmed_short': div_data.get('confirmed_short', False),
            'divergence_strength': div_data.get('divergence_strength', 0),
            'bars_since_divergence': div_data.get('bars_since_divergence'),
            'reason': entry_reason or 'No divergence signal'
        }
    }