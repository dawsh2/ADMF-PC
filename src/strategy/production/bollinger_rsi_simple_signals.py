"""
Bollinger Band + RSI Simple Signals

This strategy generates simple directional signals based on current market conditions,
without trying to track positions or holding periods.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='bollinger_rsi_simple_signals',
    feature_discovery=lambda params: [
        # Just the basics - BB and RSI
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
        FeatureSpec('rsi', {'period': params.get('rsi_period', 14)})
    ],
    parameter_space={
        'bb_period': {'type': 'int', 'range': (10, 50), 'default': 20},
        'bb_std': {'type': 'float', 'range': (1.5, 3.0), 'default': 2.0},
        'rsi_period': {'type': 'int', 'range': (10, 20), 'default': 14},
        'rsi_threshold': {'type': 'float', 'range': (5, 15), 'default': 10}
    },
    strategy_type='mean_reversion',
    tags=['bollinger_bands', 'rsi', 'mean_reversion', 'simple']
)
def bollinger_rsi_simple_signals(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Simple signal generation based on current conditions only.
    
    Signal logic:
    - Long: Price below lower band AND RSI not too oversold (shows divergence)
    - Short: Price above upper band AND RSI not too overbought (shows divergence)
    - Flat: Price near middle band (natural exit point)
    
    No position tracking, no complex state - just current market conditions.
    """
    # Get parameters
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    rsi_period = params.get('rsi_period', 14)
    rsi_threshold = params.get('rsi_threshold', 10)
    
    # Get current values
    price = bar.get('close', 0)
    middle_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_middle', price)
    upper_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_upper', price)
    lower_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_lower', price)
    rsi = features.get(f'rsi_{rsi_period}', 50)
    
    # Calculate band position
    band_width = upper_band - lower_band
    position_in_bands = (price - lower_band) / band_width if band_width > 0 else 0.5
    
    # Signal generation based on current conditions
    
    # LONG CONDITIONS: Price at lower extreme with RSI divergence
    if position_in_bands < 0:  # Below lower band
        # Check for divergence: RSI not confirming the extreme
        # The more RSI diverges from oversold, the stronger the signal
        if rsi > (30 + rsi_threshold):  # e.g., RSI > 40 when price is below band
            return {
                'signal_value': 1,
                'metadata': {
                    'reason': 'below_band_rsi_divergence',
                    'price': price,
                    'rsi': rsi,
                    'distance_below_band': (lower_band - price) / price * 100
                }
            }
    
    # SHORT CONDITIONS: Price at upper extreme with RSI divergence
    elif position_in_bands > 1:  # Above upper band
        # Check for divergence: RSI not confirming the extreme
        if rsi < (70 - rsi_threshold):  # e.g., RSI < 60 when price is above band
            return {
                'signal_value': -1,
                'metadata': {
                    'reason': 'above_band_rsi_divergence',
                    'price': price,
                    'rsi': rsi,
                    'distance_above_band': (price - upper_band) / price * 100
                }
            }
    
    # FLAT CONDITIONS: Price in middle zone
    elif 0.4 < position_in_bands < 0.6:  # Middle 20% of bands
        return {
            'signal_value': 0,
            'metadata': {
                'reason': 'middle_band_zone',
                'price': price,
                'position_in_bands': position_in_bands
            }
        }
    
    # No signal - let position ride
    return None