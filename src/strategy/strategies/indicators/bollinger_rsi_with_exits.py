"""
Bollinger Band + RSI Divergence Strategy with Proper Exits

This strategy uses the bb_rsi_dependent feature but adds logic to exit to flat
rather than always reversing positions.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='bollinger_rsi_with_exits',
    feature_discovery=lambda params: [
        # The dependent feature that tracks divergences
        FeatureSpec('bb_rsi_dependent', {
            'lookback': params.get('lookback', 20),
            'rsi_divergence_threshold': params.get('rsi_divergence_threshold', 5.0),
            'confirmation_bars': params.get('confirmation_bars', 10),
            'bb_period': params.get('bb_period', 20),
            'bb_std': params.get('bb_std', 2.0),
            'rsi_period': params.get('rsi_period', 14)
        }),
        # Bollinger bands for exit signals
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
        # RSI for exit conditions
        FeatureSpec('rsi', {'period': params.get('rsi_period', 14)})
    ],
    parameter_space={
        'bb_period': {'type': 'int', 'range': (10, 50), 'default': 20},
        'bb_std': {'type': 'float', 'range': (1.5, 3.0), 'default': 2.0},
        'rsi_period': {'type': 'int', 'range': (10, 20), 'default': 14},
        'lookback': {'type': 'int', 'range': (10, 30), 'default': 20},
        'rsi_divergence_threshold': {'type': 'float', 'range': (3.0, 10.0), 'default': 5.0},
        'confirmation_bars': {'type': 'int', 'range': (5, 20), 'default': 10},
        'min_bars_before_new_signal': {'type': 'int', 'range': (10, 50), 'default': 20},
        'exit_on_middle_band': {'type': 'bool', 'default': True},
        'exit_on_rsi_neutral': {'type': 'bool', 'default': True}
    },
    strategy_type='mean_reversion',
    tags=['divergence', 'volatility', 'momentum', 'mean_reversion', 'with_exits']
)
def bollinger_rsi_with_exits(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    BB+RSI divergence with proper exit logic.
    
    Entry: When bb_rsi_dependent confirms a divergence
    Exit: Multiple conditions that return to flat (0)
    """
    # Get parameters
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    rsi_period = params.get('rsi_period', 14)
    min_bars_before_new = params.get('min_bars_before_new_signal', 20)
    exit_on_middle = params.get('exit_on_middle_band', True)
    exit_on_rsi_neutral = params.get('exit_on_rsi_neutral', True)
    
    # Build feature key
    lookback = params.get('lookback', 20)
    rsi_divergence_threshold = params.get('rsi_divergence_threshold', 5.0)
    confirmation_bars = params.get('confirmation_bars', 10)
    
    feature_prefix = f'bb_rsi_dependent_{lookback}_{rsi_divergence_threshold}_{confirmation_bars}_{bb_period}_{bb_std}_{rsi_period}'
    
    # Get divergence signals
    confirmed_long = features.get(f'{feature_prefix}_confirmed_long', False)
    confirmed_short = features.get(f'{feature_prefix}_confirmed_short', False)
    bars_since_divergence = features.get(f'{feature_prefix}_bars_since_divergence')
    
    # Get current values
    price = bar.get('close', 0)
    middle_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_middle', price)
    upper_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_upper', price)
    lower_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_lower', price)
    rsi = features.get(f'rsi_{rsi_period}', 50)
    
    # Exit conditions first - these return to flat (0)
    if bars_since_divergence is not None and bars_since_divergence < 500:
        # Exit on middle band
        if exit_on_middle:
            band_width = upper_band - lower_band
            middle_zone = band_width * 0.15  # 15% of band width
            if abs(price - middle_band) < middle_zone:
                return {
                    'signal_value': 0,
                    'metadata': {
                        'signal_type': 'exit_middle_band',
                        'exit_price': price,
                        'bars_since_entry': bars_since_divergence
                    }
                }
        
        # Exit on RSI neutral
        if exit_on_rsi_neutral and 45 < rsi < 55:
            return {
                'signal_value': 0,
                'metadata': {
                    'signal_type': 'exit_rsi_neutral',
                    'exit_price': price,
                    'rsi': rsi,
                    'bars_since_entry': bars_since_divergence
                }
            }
        
        # Exit if held too long
        if bars_since_divergence > 200:
            return {
                'signal_value': 0,
                'metadata': {
                    'signal_type': 'exit_timeout',
                    'exit_price': price,
                    'bars_held': bars_since_divergence
                }
            }
    
    # Entry signals - only if we've been flat for a while
    if bars_since_divergence is None or bars_since_divergence > min_bars_before_new:
        if confirmed_long:
            return {
                'signal_value': 1,
                'metadata': {
                    'signal_type': 'bb_rsi_divergence_long',
                    'entry_price': price,
                    'rsi': rsi,
                    'target': middle_band
                }
            }
        elif confirmed_short:
            return {
                'signal_value': -1,
                'metadata': {
                    'signal_type': 'bb_rsi_divergence_short',
                    'entry_price': price,
                    'rsi': rsi,
                    'target': middle_band
                }
            }
    
    # Default: stay flat or maintain current position
    return None