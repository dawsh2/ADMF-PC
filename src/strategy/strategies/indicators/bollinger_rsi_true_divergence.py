"""
Bollinger Band + True RSI Divergence Strategy

This strategy uses actual RSI divergences (comparing highs/lows over time)
combined with Bollinger Band extremes for entry signals.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='bollinger_rsi_true_divergence',
    feature_discovery=lambda params: [
        # Bollinger Bands
        FeatureSpec('bollinger_bands', {
            'period': params.get('bb_period', 20),
            'std_dev': params.get('bb_std', 2.0)
        }, 'middle'),
        FeatureSpec('bollinger_bands', {
            'period': params.get('bb_period', 20),
            'std_dev': params.get('bb_std', 2.0)
        }, 'upper'),
        FeatureSpec('bollinger_bands', {
            'period': params.get('bb_period', 20),
            'std_dev': params.get('bb_std', 2.0)
        }, 'lower'),
        # RSI for current value
        FeatureSpec('rsi', {'period': params.get('rsi_period', 14)}),
        # True RSI divergence detection
        FeatureSpec('rsi_divergence', {
            'rsi_period': params.get('rsi_period', 14),
            'lookback_bars': params.get('lookback_bars', 50),
            'min_bars_between': params.get('min_bars_between', 5),
            'rsi_divergence_threshold': params.get('rsi_divergence_threshold', 5.0),
            'price_threshold_pct': params.get('price_threshold_pct', 0.001)
        })
    ],
    parameter_space={
        'bb_period': {'type': 'int', 'range': (10, 50), 'default': 20},
        'bb_std': {'type': 'float', 'range': (1.5, 3.0), 'default': 2.0},
        'rsi_period': {'type': 'int', 'range': (10, 20), 'default': 14},
        'lookback_bars': {'type': 'int', 'range': (20, 100), 'default': 50},
        'min_bars_between': {'type': 'int', 'range': (5, 20), 'default': 5},
        'rsi_divergence_threshold': {'type': 'float', 'range': (3.0, 10.0), 'default': 5.0},
        'price_threshold_pct': {'type': 'float', 'range': (0.0005, 0.005), 'default': 0.001},
        'require_band_extreme': {'type': 'bool', 'default': True},
        'exit_at_middle': {'type': 'bool', 'default': True}
    },
    strategy_type='mean_reversion',
    tags=['divergence', 'volatility', 'momentum', 'mean_reversion', 'true_divergence']
)
def bollinger_rsi_true_divergence(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    True RSI divergence strategy.
    
    Entry: When true RSI divergence is detected, optionally at BB extremes
    Exit: When price returns to middle band zone
    """
    # Get parameters
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    rsi_period = params.get('rsi_period', 14)
    require_band_extreme = params.get('require_band_extreme', True)
    exit_at_middle = params.get('exit_at_middle', True)
    
    # Get current values
    price = bar.get('close', 0)
    middle_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_middle', price)
    upper_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_upper', price)
    lower_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_lower', price)
    rsi = features.get(f'rsi_{rsi_period}', 50)
    
    # Get divergence signals from the features
    # The feature hub stores multi-component features with sub-keys
    lookback = params.get('lookback_bars', 50)
    min_bars = params.get('min_bars_between', 5)
    rsi_thresh = params.get('rsi_divergence_threshold', 5.0)
    price_thresh = params.get('price_threshold_pct', 0.001)
    
    # Build the base feature key
    div_base_key = f'rsi_divergence_{lookback}_{min_bars}_{price_thresh}_{rsi_thresh}_{rsi_period}'
    
    # Get individual divergence components with sub-keys
    has_bullish_div = features.get(f'{div_base_key}_has_bullish_divergence', False)
    has_bearish_div = features.get(f'{div_base_key}_has_bearish_divergence', False)
    div_strength = features.get(f'{div_base_key}_divergence_strength', 0)
    bars_since_div = features.get(f'{div_base_key}_bars_since_divergence')
    
    # Calculate band position
    band_width = upper_band - lower_band
    position_in_bands = (price - lower_band) / band_width if band_width > 0 else 0.5
    
    # Debug: Log when divergences are detected
    if has_bullish_div or has_bearish_div:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"DIVERGENCE DETECTED! Bullish: {has_bullish_div}, Bearish: {has_bearish_div}, Strength: {div_strength}")
        logger.warning(f"Price: {price}, Lower: {lower_band}, Upper: {upper_band}, Position: {position_in_bands}")
        logger.warning(f"Require band extreme: {require_band_extreme}")
    
    # Entry conditions - true divergence detected
    if has_bullish_div:
        # For bullish divergence, enter when price is OUTSIDE (below) lower band
        if not require_band_extreme or price < lower_band:
            return {
                'signal_value': 1,
                'metadata': {
                    'signal_type': 'true_bullish_divergence',
                    'divergence_strength': div_strength,
                    'entry_price': price,
                    'rsi': rsi,
                    'position_in_bands': position_in_bands,
                    'outside_bands': True
                }
            }
    
    elif has_bearish_div:
        # For bearish divergence, enter when price is OUTSIDE (above) upper band
        if not require_band_extreme or price > upper_band:
            return {
                'signal_value': -1,
                'metadata': {
                    'signal_type': 'true_bearish_divergence',
                    'divergence_strength': div_strength,
                    'entry_price': price,
                    'rsi': rsi,
                    'position_in_bands': position_in_bands,
                    'outside_bands': True
                }
            }
    
    # Exit conditions - return to middle band
    if exit_at_middle and bars_since_div and bars_since_div < 100:
        # Exit zone: middle 20% of bands
        if 0.4 < position_in_bands < 0.6:
            return {
                'signal_value': 0,
                'metadata': {
                    'signal_type': 'exit_at_middle',
                    'exit_price': price,
                    'position_in_bands': position_in_bands,
                    'bars_since_entry': bars_since_div
                }
            }
    
    # No signal
    return None