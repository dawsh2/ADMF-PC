"""
Bollinger Band + RSI Divergence Strategy - Fixed Exit Logic

This version properly implements exits at middle band without the 20-bar restriction.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='bollinger_rsi_dependent_fixed',
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
        # RSI is computed as a dependency of bb_rsi_dependent
        FeatureSpec('rsi', {'period': params.get('rsi_period', 14)})
    ],
    parameter_space={
        'bb_period': {'type': 'int', 'range': (10, 50), 'default': 20},
        'bb_std': {'type': 'float', 'range': (1.5, 3.0), 'default': 2.0},
        'rsi_period': {'type': 'int', 'range': (10, 20), 'default': 14},
        'lookback': {'type': 'int', 'range': (10, 30), 'default': 20},
        'rsi_divergence_threshold': {'type': 'float', 'range': (3.0, 10.0), 'default': 5.0},
        'confirmation_bars': {'type': 'int', 'range': (5, 20), 'default': 10},
        'exit_at_middle_band': {'type': 'bool', 'default': True},
        'max_hold_bars': {'type': 'int', 'range': (50, 500), 'default': 200},
        'quick_exit_bars': {'type': 'int', 'range': (10, 50), 'default': 30}
    },
    strategy_type='mean_reversion',
    tags=['divergence', 'volatility', 'momentum', 'mean_reversion', 'multi_bar_pattern']
)
def bollinger_rsi_dependent_fixed(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Fixed version with proper exit logic.
    
    Entry: When bb_rsi_dependent confirms a divergence
    Exit: At middle band, max hold time, or new opposite signal
    """
    # Get parameters
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    exit_at_middle_band = params.get('exit_at_middle_band', True)
    max_hold_bars = params.get('max_hold_bars', 200)
    quick_exit_bars = params.get('quick_exit_bars', 30)
    
    # Get divergence signals
    lookback = params.get('lookback', 20)
    rsi_divergence_threshold = params.get('rsi_divergence_threshold', 5.0)
    confirmation_bars = params.get('confirmation_bars', 10)
    rsi_period = params.get('rsi_period', 14)
    
    feature_prefix = f'bb_rsi_dependent_{lookback}_{rsi_divergence_threshold}_{confirmation_bars}_{bb_period}_{bb_std}_{rsi_period}'
    
    confirmed_long = features.get(f'{feature_prefix}_confirmed_long', False)
    confirmed_short = features.get(f'{feature_prefix}_confirmed_short', False)
    bars_since_divergence = features.get(f'{feature_prefix}_bars_since_divergence')
    
    # Get current price and bands for exit logic
    price = bar.get('close', 0)
    middle_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_middle', price)
    upper_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_upper', price)
    lower_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_lower', price)
    
    # Track position state (this is a limitation of stateless strategies)
    # We infer position from recent divergence signals and price location
    in_long_position = (bars_since_divergence is not None and 
                       bars_since_divergence < max_hold_bars and
                       features.get(f'{feature_prefix}_has_bullish_divergence', False))
    
    in_short_position = (bars_since_divergence is not None and 
                        bars_since_divergence < max_hold_bars and
                        features.get(f'{feature_prefix}_has_bearish_divergence', False))
    
    # Exit signals with multiple conditions
    if exit_at_middle_band and (in_long_position or in_short_position):
        # Quick exit if price reaches middle band soon after entry
        if bars_since_divergence < quick_exit_bars:
            if in_long_position and price >= middle_band:
                return {
                    'signal_value': 0,
                    'metadata': {
                        'signal_type': 'quick_exit_at_middle',
                        'exit_price': price,
                        'bars_held': bars_since_divergence
                    }
                }
            elif in_short_position and price <= middle_band:
                return {
                    'signal_value': 0,
                    'metadata': {
                        'signal_type': 'quick_exit_at_middle',
                        'exit_price': price,
                        'bars_held': bars_since_divergence
                    }
                }
        
        # Standard exit at middle band (for positions held longer)
        elif bars_since_divergence < max_hold_bars:
            # For longs: exit if price crosses above middle band
            if in_long_position and price >= middle_band:
                return {
                    'signal_value': 0,
                    'metadata': {
                        'signal_type': 'exit_at_middle',
                        'exit_price': price,
                        'bars_held': bars_since_divergence
                    }
                }
            # For shorts: exit if price crosses below middle band
            elif in_short_position and price <= middle_band:
                return {
                    'signal_value': 0,
                    'metadata': {
                        'signal_type': 'exit_at_middle',
                        'exit_price': price,
                        'bars_held': bars_since_divergence
                    }
                }
    
    # Time-based exit (max hold period reached)
    if bars_since_divergence and bars_since_divergence >= max_hold_bars:
        if in_long_position or in_short_position:
            return {
                'signal_value': 0,
                'metadata': {
                    'signal_type': 'max_hold_exit',
                    'exit_price': price,
                    'bars_held': bars_since_divergence
                }
            }
    
    # Entry signals (only if not in position or after exit)
    if confirmed_long and not in_long_position:
        print(f"[STRATEGY] Generating LONG signal at bar {bar.get('timestamp', 0)}")
        return {
            'signal_value': 1,
            'metadata': {
                'signal_type': 'bb_rsi_divergence_long',
                'has_divergence': features.get(f'{feature_prefix}_has_bullish_divergence', False),
                'divergence_strength': features.get(f'{feature_prefix}_divergence_strength', 0.0),
                'entry_price': price,
                'target_price': middle_band
            }
        }
    elif confirmed_short and not in_short_position:
        print(f"[STRATEGY] Generating SHORT signal at bar {bar.get('timestamp', 0)}")
        return {
            'signal_value': -1,
            'metadata': {
                'signal_type': 'bb_rsi_divergence_short',
                'has_divergence': features.get(f'{feature_prefix}_has_bearish_divergence', False),
                'divergence_strength': features.get(f'{feature_prefix}_divergence_strength', 0.0),
                'entry_price': price,
                'target_price': middle_band
            }
        }
    
    # No signal
    return None